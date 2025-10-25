import torch 

def train_unsupervised(model, patterns, n_epochs=1, q_list=None, lr_list=None):
        n_s = len(model.slayers)
        q_list = q_list or [1.0] * n_s
        lr_list = lr_list or [1.0] * n_s

        for ep in range(n_epochs):
            idx = torch.randperm(patterns.shape[0])
            for i in idx: #Each pattern passes through full network
                p = patterns[i:i+1]
                activations, _ = model.forward(p)
                sl_idx = 0
                u_prev, v_prev = None, None
                for typ, out, patches, layer in activations:
                    if typ == 'S':
                        # Representative selection
                        B, P, H, W = out.shape
                        flattened = out.view(P, H * W) #flatten the spatial map of S-layer
                        col_vals, col_idx = torch.max(flattened, dim=0) #find max response across planes
                        reps = []
                        patches_1d = patches[0]
                        for plane in range(P): #for each S-plane select the strongest responding S-cell in overlapping columns. Ensures each S-plane specializes in a different feature.
                            mask = (col_idx == plane)
                            if mask.sum() == 0:
                                continue
                            vals = flattened[plane] * mask.float()
                            mv, mp = torch.max(vals, dim=0)
                            if mv.item() > 0:
                                patch_vec = patches_1d[:, mp].detach()
                                reps.append((plane, int(mp.item()), patch_vec))
                        if reps and u_prev is not None and v_prev is not None: #Calls the reinforce method on representative S-cells.
                            c_mask = torch.ones(layer.rf, layer.rf)
                            layer.reinforce(reps, u_prev[0], v_prev[0].mean(0), c_mask, q_l=q_list[sl_idx])
                        sl_idx += 1
                    elif typ == 'C':
                        u_prev = out
                        v_prev = out  # simplified inhibitory analog
            print(f"Epoch {ep+1}/{n_epochs} complete.")