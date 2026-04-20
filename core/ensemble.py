

# ============================================================================
# 1. Basic class definitions
# ============================================================================

class HFTextEmbedder(nn.Module):
    """HuggingFace text embedder"""
    def __init__(self, model_name: str, device: torch.device, local_files_only: bool = False): 
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.encoder.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.encoder(**toks)
        # Mean Pooling
        last = out.last_hidden_state
        attn = toks['attention_mask']
        mask = attn.unsqueeze(-1).expand(last.size()).float()
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)
        return F.normalize(emb, p=2, dim=-1)

class LowRankSemanticAdapter(nn.Module):
    """
    Low-Rank Semantic Adapter
    """
    def __init__(self,
                 unified_model_names: List[str],
                 target_model_names: List[str],
                 text_embedder_name: str,
                 device: torch.device,
                 temperature: float = 1.0,
                 local_files_only: bool = True,
                 num_routers: int = 0,
                 embedding_dim: int = 384,
                 rank: int = 16,
                 **kwargs
                 ):
        super().__init__()
        self.device = device
        self.num_unified = len(unified_model_names)
        self.num_target = len(target_model_names)
        
        print(f"Initializing LowRankSemanticAdapter: {num_routers} Routers -> {self.num_target} Targets")
        print(f"  | Semantic Correction Rank: {rank} | Embedding Dim: {embedding_dim}")

        self.embedder = HFTextEmbedder(text_embedder_name, device, local_files_only=local_files_only)
        for param in self.embedder.parameters():
            param.requires_grad = False 

        if num_routers > 0:
            self.router_weights = nn.Parameter(torch.zeros(num_routers, device=device))
        else:
            self.router_weights = None

        with torch.no_grad():
            emb_u = self.embedder.encode_texts(unified_model_names)
            emb_l = self.embedder.encode_texts(target_model_names)
            sim = torch.matmul(emb_u, emb_l.T)
            init_W = F.softmax(sim / temperature, dim=0)
        
        self.register_buffer('projection_matrix', init_W)

        self.semantic_encoder = nn.Sequential(
            nn.Linear(embedding_dim, rank, bias=False),
            nn.LayerNorm(rank),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(rank, self.num_target, bias=False)
        )
        
        nn.init.normal_(self.semantic_encoder[0].weight, std=0.01)
        nn.init.zeros_(self.semantic_encoder[-1].weight)

        self.beta = nn.Parameter(torch.zeros(self.num_target, device=device))

    def _normalize_router_logits(self, stack, mask):
        eps = 1e-6
        mask_f = mask.float()
        valid_cnt = mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        masked_val = stack * mask_f
        mean = masked_val.sum(dim=2, keepdim=True) / valid_cnt
        var = ((stack - mean)**2 * mask_f).sum(dim=2, keepdim=True) / valid_cnt
        std = torch.sqrt(var + eps)
        normalized = (stack - mean) / std
        return normalized * mask_f

    def get_router_weights(self):
        if self.router_weights is None: return None
        return F.softmax(self.router_weights, dim=0)

    def forward(self, aligned_stack: torch.Tensor, mask_stack: torch.Tensor, prompts: List[str] = None, input_embedding: torch.Tensor = None) -> torch.Tensor:
        B, R, U = aligned_stack.shape
        if input_embedding is None:
            if prompts is None:
                raise ValueError("Must provide either prompts or input_embedding")
            with torch.no_grad():
                input_embedding = self.embedder.encode_texts(prompts)
        
        norm_stack = self._normalize_router_logits(aligned_stack, mask_stack)
        projected_stack = torch.matmul(norm_stack, self.projection_matrix)
        
        if self.router_weights is not None:
            w = self.get_router_weights().view(1, R, 1)
            router_valid = mask_stack.sum(dim=2) > 0
            router_valid_weight = router_valid.float() * w.squeeze(-1)
            router_valid_weight = router_valid_weight / router_valid_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
            router_valid_weight = router_valid_weight.unsqueeze(-1)
            base_logits = (projected_stack * router_valid_weight).sum(dim=1)
        else:
            base_logits = projected_stack.mean(dim=1)

        delta_logits = self.semantic_encoder(input_embedding)
        delta_scale = 1.0
        final_logits = base_logits + delta_logits*delta_scale + self.beta
        return final_logits

    def forward_with_router_stack(self, aligned_stack: torch.Tensor, mask_stack: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        return self.forward(aligned_stack, mask_stack, prompts=prompts)

@dataclass
class OnlineEnsembleTrainingSample:
    question_id: Any
    prompt: str
    target_logits: torch.Tensor 
    target_id: Optional[int] = None
    type: Optional[str] = None

class OnlineEnsembleSampleDataset(Dataset):
    def __init__(self, samples: List[OnlineEnsembleTrainingSample]):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_online_ensemble_samples(batch: List[OnlineEnsembleTrainingSample]) -> Dict[str, Any]:
    target_logits = torch.stack([item.target_logits for item in batch], dim=0)
    prompts = [item.prompt for item in batch]
    types = [item.type if item.type is not None else "general" for item in batch]
    return {
        "target_logits": target_logits, 
        "prompts": prompts,
        "type": types 
    }

# ============================================================================
# 2. ✅ [Core modification] Black-box data loading and processing
# ============================================================================

def load_blackbox_data_multi_file(
    file_paths: List[str],
    smoothing_factor: float = 0.1,
    seed: int = 42,
    train_split: float = 0.8,
    size: int = 1200 # Target total sample count
) -> Tuple[List[OnlineEnsembleTrainingSample], List[OnlineEnsembleTrainingSample], List[str]]:
    """
    Load data from multiple black-box log files, automatically discover all models,
    apply label smoothing, and extract dataset_name from file attributes as type.
    """
    
    all_records = []
    
    # --- 1. Load all files ---
    print(f"\n📚 [BlackBox Loader] Loading from {len(file_paths)} files...")
    
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"⚠️ Warning: File not found: {path}")
            continue
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                recs = []
                file_dataset_name = "blackbox_unknown" 

                if isinstance(data, dict):
                    # Try to get dataset name as type
                    if "dataset_name" in data:
                        file_dataset_name = data["dataset_name"]
                    elif "dataset" in data:
                        file_dataset_name = data["dataset"]
                    else:
                        file_dataset_name = path.stem 
                    
                    if "records" in data:
                        recs = data["records"]
                        
                elif isinstance(data, list):
                    recs = data
                    file_dataset_name = path.stem 
                
                if not recs:
                    print(f"⚠️ Warning: No records in {path.name}")
                    continue

                # Inject dataset_name into each record
                for r in recs:
                    r['_source_dataset'] = file_dataset_name
                    r['_source_file'] = path.name
                
                all_records.extend(recs)
                print(f"   - Loaded {len(recs)} records from {path.name} (Type: {file_dataset_name})")
                
        except Exception as e:
            print(f"❌ Error reading {path.name}: {e}")

    if not all_records:
        raise ValueError("No valid records found in any provided files.")

    # --- 2. Auto-discover all target models ---
    discovered_models = set()
    for item in all_records:
        model_name = None
        extra = item.get("extra_fields", {})
        if extra: model_name = extra.get("actual_model")
        if not model_name: model_name = item.get("model_name")
        if model_name: discovered_models.add(model_name)
    
    target_model_names = sorted(list(discovered_models))
    model_to_idx = {name: i for i, name in enumerate(target_model_names)}
    num_classes = len(target_model_names)
    
    print(f"✅ Discovered {num_classes} unique models: {target_model_names}")

    # --- 3. Construct samples (Label Smoothing) ---
    raw_samples = []
    
    if num_classes > 1:
        prob_target = 1.0 - smoothing_factor
        prob_others = smoothing_factor / (num_classes - 1)
    else:
        prob_target, prob_others = 1.0, 0.0

    global_idx = 0
    
    for item in all_records:
        prompt = item.get("prompt") or item.get("origin_query")
        if not prompt: continue

        extra = item.get("extra_fields", {})
        chosen_model = extra.get("actual_model") or item.get("model_name")
        
        if chosen_model not in model_to_idx: continue
        target_idx = model_to_idx[chosen_model]

        # Construct smoothed distribution
        probs = torch.full((num_classes,), prob_others, dtype=torch.float32)
        probs[target_idx] = prob_target
        # ⚠️ Must take log, since loss function expects logits (for later softmax restoration)
        target_logits = torch.log(probs + 1e-9)

        # Extract previously injected dataset_name
        sample_type = item.get('_source_dataset', 'blackbox_generic')

        raw_samples.append(OnlineEnsembleTrainingSample(
            question_id=f"bb_{global_idx}", 
            prompt=prompt,
            target_logits=target_logits,
            target_id=target_idx,
            type=sample_type 
        ))
        global_idx += 1

    # --- 4. Statistics and balancing ---
    print(f"\n📊 [Statistics] Raw Samples Distribution (Total: {len(raw_samples)}):")
    raw_counts = defaultdict(int)
    for s in raw_samples: raw_counts[s.target_id] += 1
    for idx, name in enumerate(target_model_names):
        count = raw_counts[idx]
        ratio = count / len(raw_samples) if len(raw_samples) > 0 else 0
        print(f"   - {name:<30}: {count:5d} ({ratio:.2%})")

    # Category balancing logic
    random.seed(seed)
    random.shuffle(raw_samples)

    if size is not None and size > 0:
        print(f"\n⚖️  [Balancing] Target Total Size: {size}")
        idx_count = defaultdict(int)
        balanced_samples = []
        max_per_idx = size // num_classes
        if max_per_idx == 0: max_per_idx = 1

        # Pass 1: Fill up to quota
        for sample in raw_samples:
            tid = sample.target_id
            if idx_count[tid] < max_per_idx:
                balanced_samples.append(sample)
                idx_count[tid] += 1
            if len(balanced_samples) >= size: break
        
        # Pass 2: Fill remaining
        if len(balanced_samples) < size:
            print(f"   ⚠️ Initial balance reached {len(balanced_samples)}, filling remaining...")
            selected_ids = set(s.question_id for s in balanced_samples)
            for sample in raw_samples:
                if len(balanced_samples) >= size: break
                if sample.question_id not in selected_ids:
                    balanced_samples.append(sample)
                    selected_ids.add(sample.question_id)
                    idx_count[sample.target_id] += 1

        final_samples = balanced_samples
    else:
        final_samples = raw_samples

    print(f"✅ Balanced Samples (Actual: {len(final_samples)}):")
    bal_counts = defaultdict(int)
    for s in final_samples: bal_counts[s.target_id] += 1
    for idx, name in enumerate(target_model_names):
        print(f"   - {name:<30}: {bal_counts[idx]:5d}")

    random.shuffle(final_samples)
    split_idx = int(len(final_samples) * train_split)
    
    return final_samples[:split_idx], final_samples[split_idx:], target_model_names

# ============================================================================
# 3. Helper functions (Loss & Alignment)
# ============================================================================

def _collect_router_stack(
    prompts: List[str],
    router_pairs: List[Any],
    unified_model_names: List[str],
    device: torch.device,
    types: Optional[List[str]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect all router outputs and align to unified space"""
    B = len(prompts)
    U = len(unified_model_names)
    R = len(router_pairs)
    u_map = {n: i for i, n in enumerate(unified_model_names)}

    aligned_stack = torch.zeros((B, R, U), device=device)
    mask_stack = torch.zeros((B, R, U), dtype=torch.bool, device=device)

    with torch.no_grad():
        for r_idx, pair in enumerate(router_pairs):
            # Note: type here is passed to router for adapter selection (if router supports it)
            raw_logits_list = [
                pair.router.route(p, suffix="", type=(types[i] if types else None))
                for i, p in enumerate(prompts)
            ]
            raw_logits = torch.stack(raw_logits_list, dim=0).to(device)
            router_models = pair.router.get_model_list()
            for local_idx, m in enumerate(router_models):
                if m in u_map:
                    u_idx = u_map[m]
                    aligned_stack[:, r_idx, u_idx] = raw_logits[:, local_idx]
                    mask_stack[:, r_idx, u_idx] = True

    return aligned_stack, mask_stack

def compute_kl_loss(
    pred_logits: torch.Tensor,
    target_logits: torch.Tensor,
    label_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
    target_probs = F.softmax(target_logits / label_temperature, dim=-1)
    
    kl_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
    
    loss_dict = {'total': kl_loss.item(), 'kl': kl_loss.item()}
    return kl_loss, loss_dict

# ============================================================================
# 4. ✅ [Core modification] Training and validation flow (Black-box mode)
# ============================================================================

def train_semantic_adapter_online(
    adapter: LowRankSemanticAdapter,
    train_samples: List[OnlineEnsembleTrainingSample],
    ensemble_router_pairs: List[Any],
    unified_model_names: List[str],
    target_model_names: List[str],
    test_samples: Optional[List[OnlineEnsembleTrainingSample]] = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.03,
    label_temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
    text_embedder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    patience: int = 10
):
    print("="*80)
    print(f"Starting Semantic Adapter training | Mode: BLACK-BOX (KL Loss)")
    print("="*80)

    train_loader = DataLoader(
        OnlineEnsembleSampleDataset(train_samples),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate_online_ensemble_samples
    )
    if test_samples:
        test_loader = DataLoader(
            OnlineEnsembleSampleDataset(test_samples),
            batch_size=batch_size,
            collate_fn=collate_online_ensemble_samples
        )

    optimizer = optim.Adam(adapter.parameters(), lr=lr)
    adapter.train()
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None
    
    for epoch in range(epochs):
        total_loss, steps = 0.0, 0
        total_kl = 0.0
        
        for batch in train_loader:
            prompts = batch["prompts"]
            types = batch["type"] 
            target_logits = batch["target_logits"].to(device).squeeze(1)

            aligned_stack, mask_stack = _collect_router_stack(
                prompts, ensemble_router_pairs, unified_model_names, device, types
            )
            pred_logits = adapter.forward_with_router_stack(aligned_stack, mask_stack, prompts=prompts)

            # Black-box mode: use KL divergence loss
            loss, loss_dict = compute_kl_loss(
                pred_logits, target_logits,
                label_temperature=label_temperature
            )
            total_kl += loss_dict.get("kl", 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | KL: {total_kl/steps:.4f}")
        
        if test_samples:
            cur_val_loss, cur_val_acc = validate(
                adapter, test_loader, ensemble_router_pairs, 
                unified_model_names, device, label_temperature
            )
            
            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                best_state_dict = adapter.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"   🎯 Best Model! Val Loss: {best_val_loss:.4f} (Acc: {cur_val_acc:.2%})")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n⏹️ Early Stopping at Epoch {best_epoch}")
                break
    
    if best_state_dict is not None:
        adapter.load_state_dict(best_state_dict)
        print(f"\n✅ Loaded Best Model (Epoch {best_epoch})")
    
    return adapter

@torch.no_grad()
def validate(
    adapter, loader, pairs, u_names, device, 
    label_temperature=1.0
):
    adapter.eval()
    total_loss = 0
    correct_top1 = 0
    total_samples = 0
    
    for batch in loader:
        aligned_stack, mask_stack = _collect_router_stack(
            batch["prompts"], pairs, u_names, device, batch["type"]
        )
        pred = adapter.forward_with_router_stack(aligned_stack, mask_stack, prompts=batch["prompts"])
        target = batch["target_logits"].to(device).squeeze(1)
        
        # Black-box mode: use KL divergence loss
        loss, _ = compute_kl_loss(
            pred, target,
            label_temperature=label_temperature
        )
        
        total_loss += loss.item()
        
        pred_ids = pred.argmax(dim=-1)
        target_ids = target.argmax(dim=-1)
        correct_top1 += (pred_ids == target_ids).sum().item()
        total_samples += target.size(0)
        
    acc = correct_top1 / total_samples
    print(f">> Val Loss: {total_loss/len(loader):.4f} | Top-1 Acc: {acc*100:.2f}%")
    
    adapter.train()
    return total_loss / len(loader), acc

def save_adapter_checkpoint(
    adapter: nn.Module,
    save_path: Union[str, Path],
    unified_model_names: List[str],
    target_model_names: List[str],
    router_names: List[str],
    history: Dict = None,
    text_embedder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        embedding_dim = adapter.semantic_encoder[0].in_features
        rank = adapter.semantic_encoder[0].out_features
    except:
        embedding_dim = 384
        rank = 16
    
    num_routers = 0
    if hasattr(adapter, 'router_weights') and adapter.router_weights is not None:
        num_routers = adapter.router_weights.shape[0]
    
    raw_state_dict = adapter.state_dict()
    clean_state_dict = {k: v for k, v in raw_state_dict.items() if not k.startswith("embedder.")}
    
    checkpoint = {
        "state_dict": clean_state_dict,
        "unified_model_names": unified_model_names,
        "target_model_names": target_model_names,
        "router_names": router_names,
        "history": history or {},
        "config": {
            "text_embedder_name": text_embedder_name,
            "embedding_dim": embedding_dim,
            "rank": rank,
            "num_routers": num_routers,
            "adapter_class": adapter.__class__.__name__
        }
    }
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved to {save_path}")

# ============================================================================
# 5. Main (Logic modification)
# ============================================================================

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Load source system (Unified Space)
    print(f"\n{'='*40}\nLoading Source Routers...\n{'='*40}")
    ensemble_pairs, _ = load_ensemble_system(args) # No longer need target_router_pair since target comes from file
    
    # Build unified space
    all_source_models = set()
    for pair in ensemble_pairs:
        all_source_models.update(pair.router.get_model_list())
    unified_model_names_U = sorted(list(all_source_models))
    print(f"Unified Space Size: {len(unified_model_names_U)}")

    # 2. ✅ Get black-box file paths
    # Assume args has blackbox_path, supports single string or list
    blackbox_files = []
    if hasattr(args, 'blackbox_data_path') and args.blackbox_data_path:
        if isinstance(args.blackbox_data_path, list):
            blackbox_files = args.blackbox_data_path
        elif isinstance(args.blackbox_data_path, str):
            blackbox_files = args.blackbox_data_path.split(',') if ',' in args.blackbox_data_path else [args.blackbox_data_path]
    
    if not blackbox_files:
        print("❌ Error: No blackbox files provided (args.blackbox_path is empty).")
        return

    # 3. ✅ Load black-box data (train/test sets and discovered models)
    print(f"\n{'='*40}\nLoading Black-box Data...\n{'='*40}")
    train_samples, test_samples, target_models_L = load_blackbox_data_multi_file(
        file_paths=blackbox_files,
        smoothing_factor=0.2, # Black-box mode recommends 0.2
        seed=args.seed,
        size=getattr(args, 'dataset_size', 150) # Default 1200 if dataset_size not in args
    )
    
    if not target_models_L:
        print("❌ Error: No models discovered from files.")
        return

    # 4. Initialize adapter
    print("\n" + "="*80)
    print("Initializing LowRankSemanticAdapter (Black-box Mode)")
    print(f"   Target Space Size: {len(target_models_L)}")
    print("="*80)

    adapter = LowRankSemanticAdapter(
        unified_model_names=unified_model_names_U,
        target_model_names=target_models_L, # Use discovered model list
        text_embedder_name=args.embedder_name,
        device=device,
        local_files_only=True,
        num_routers=len(ensemble_pairs),
        embedding_dim=args.embedding_dim,
        rank=args.rank
    ).to(device)

    # 5. Start training with black-box hybrid loss
    history = {}
    if ensemble_pairs:
        history = train_semantic_adapter_online(
            adapter=adapter,
            train_samples=train_samples,
            ensemble_router_pairs=ensemble_pairs,
            unified_model_names=unified_model_names_U,
            target_model_names=target_models_L,
            test_samples=test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )

    # 6. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{args.target_router_name}_blackbox_adapter.pt"
    
    save_adapter_checkpoint(
        adapter=adapter,
        save_path=save_path,
        unified_model_names=unified_model_names_U,
        target_model_names=target_models_L,
        router_names=[p.name for p in ensemble_pairs],
        history=history,
        text_embedder_name=args.embedder_name
    )

if __name__ == "__main__":
    main()