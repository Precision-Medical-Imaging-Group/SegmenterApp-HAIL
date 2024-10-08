
import numpy as np
import torch
import os

from datetime import datetime
import nibabel as nib

from utils import reparameterize_logit, divide_into_batches

from network import UNet, ThetaEncoder, EtaEncoder, Patchifier, AttentionModule


class HAIL:
    def __init__(self, beta_dim, theta_dim, eta_dim, pretrained=None, pretrained_eta_encoder=None, gpu_id=0):
        self.beta_dim = beta_dim
        self.theta_dim = theta_dim
        self.eta_dim = eta_dim
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.timestr = datetime.now().strftime("%Y%m%d-%H%M%S")


        # define networks
        self.beta_encoder = UNet(in_ch=1, out_ch=self.beta_dim, base_ch=8, final_act='none')
        self.theta_encoder = ThetaEncoder(in_ch=1, out_ch=self.theta_dim)
        self.eta_encoder = EtaEncoder(in_ch=1, out_ch=self.eta_dim)
        self.attention_module = AttentionModule(self.theta_dim + self.eta_dim, v_ch=self.beta_dim)
        self.decoder = UNet(in_ch=1 + self.theta_dim, out_ch=1, base_ch=16, final_act='relu')
        self.patchifier = Patchifier(in_ch=1, out_ch=128)

        if pretrained_eta_encoder is not None:
            checkpoint_eta_encoder = torch.load(pretrained_eta_encoder, map_location=self.device)
            self.eta_encoder.load_state_dict(checkpoint_eta_encoder['eta_encoder'])
        if pretrained is not None:
            self.checkpoint = torch.load(pretrained, map_location=self.device)
            self.beta_encoder.load_state_dict(self.checkpoint['beta_encoder'])
            self.theta_encoder.load_state_dict(self.checkpoint['theta_encoder'])
            self.eta_encoder.load_state_dict(self.checkpoint['eta_encoder'])
            self.decoder.load_state_dict(self.checkpoint['decoder'])
            self.attention_module.load_state_dict(self.checkpoint['attention_module'])
            self.patchifier.load_state_dict(self.checkpoint['patchifier'])
        self.beta_encoder.to(self.device)
        self.theta_encoder.to(self.device)
        self.eta_encoder.to(self.device)
        self.decoder.to(self.device)
        self.attention_module.to(self.device)
        self.patchifier.to(self.device)

    def channel_aggregation(self, beta_onehot_encode):

        
        
        """
        Combine multi-channel one-hot encoded beta into one channel (label-encoding).

        ===INPUTS===
        * beta_onehot_encode: torch.Tensor (batch_size, self.beta_dim, image_dim, image_dim)
            One-hot encoded beta variable. At each pixel location, only one channel will take value of 1,
            and other channels will be 0.
        ===OUTPUTS===
        * beta_label_encode: torch.Tensor (batch_size, 1, image_dim, image_dim)
            The intensity value of each pixel will be determined by the channel index with value of 1.
        """
        batch_size = beta_onehot_encode.shape[0]
        image_dim = beta_onehot_encode.shape[3]
        value_tensor = (torch.arange(0, self.beta_dim) * 1.0).to(self.device)
        value_tensor = value_tensor.view(1, self.beta_dim, 1, 1).repeat(batch_size, 1, image_dim, image_dim)
        beta_label_encode = beta_onehot_encode * value_tensor.detach()
        return beta_label_encode.sum(1, keepdim=True) / self.beta_dim


    def harmonize(self, source_images, target_images, target_theta, target_eta, out_paths,
                  recon_orientation, norm_vals, header=None, num_batches=4, save_intermediate=False, intermediate_out_dir=None):
        if out_paths is not None:
            for out_path in out_paths:
                os.makedirs(out_path.parent, exist_ok=True)
            prefix = str(out_paths[0].name).split('.')[0]
        
        if save_intermediate:
            os.makedirs(intermediate_out_dir, exist_ok=True)

        with torch.set_grad_enabled(False):
            self.beta_encoder.eval()
            self.theta_encoder.eval()
            self.eta_encoder.eval()
            self.decoder.eval()

            # === 1. CALCULATE BETA, THETA, ETA FROM SOURCE IMAGES ===
            logits, betas, keys, masks = [], [], [], []
            for source_image in source_images:
                source_image = source_image.unsqueeze(1)
                source_image_batches = divide_into_batches(source_image, num_batches)
                mask_tmp, logit_tmp, beta_tmp, key_tmp = [], [], [], []
                for source_image_batch in source_image_batches:
                    batch_size = source_image_batch.shape[0]
                    source_image_batch = source_image_batch.to(self.device)
                    mask = (source_image_batch > 1e-6) * 1.0
                    logit = self.beta_encoder(source_image_batch)
                    beta = self.channel_aggregation(reparameterize_logit(logit))
                    theta_source, _ = self.theta_encoder(source_image_batch)
                    eta_source = self.eta_encoder(source_image_batch).view(batch_size, self.eta_dim, 1, 1)
                    mask_tmp.append(mask)
                    logit_tmp.append(logit)
                    beta_tmp.append(beta)
                    key_tmp.append(torch.cat([theta_source, eta_source], dim=1))
                masks.append(torch.cat(mask_tmp, dim=0))
                logits.append(torch.cat(logit_tmp, dim=0))
                betas.append(torch.cat(beta_tmp, dim=0))
                keys.append(torch.cat(key_tmp, dim=0))

            # === 2. CALCULATE THETA, ETA FOR TARGET IMAGES (IF NEEDED) ===
            if target_theta is None:
                queries, thetas_target = [], []
                for target_image in target_images:
                    target_image = target_image.to(self.device).unsqueeze(1)
                    theta_target, _ = self.theta_encoder(target_image)
                    theta_target = theta_target.mean(dim=0, keepdim=True)
                    eta_target = self.eta_encoder(target_image).mean(dim=0, keepdim=True).view(1, self.eta_dim, 1, 1)
                    thetas_target.append(theta_target)
                    queries.append(
                        torch.cat([theta_target, eta_target], dim=1).view(1, self.theta_dim + self.eta_dim, 1))
                if save_intermediate:
                    # save theta and eta of target images
                    with open(intermediate_out_dir / f'{prefix}_targets.txt', 'w') as fp:
                        fp.write(','.join(['img'] + [f'theta{i}' for i in range(self.theta_dim)] +
                                          [f'eta{i}' for i in range(self.eta_dim)]) + '\n')
                        for i, img_query in enumerate([query.squeeze().cpu().numpy().tolist() for query in queries]):
                            fp.write(','.join([f'target{i}'] + ['%.6f' % val for val in img_query]) + '\n')
            else:
                queries, thetas_target = [], []
                for target_theta_tmp, target_eta_tmp in zip(target_theta, target_eta):
                    thetas_target.append(target_theta_tmp.view(1, self.theta_dim, 1, 1).to(self.device))
                    queries.append(torch.cat([target_theta_tmp.view(1, self.theta_dim, 1).to(self.device),
                                              target_eta_tmp.view(1, self.eta_dim, 1).to(self.device)], dim=1))

            # === 3. SAVE ENCODED VARIABLES (IF REQUESTED) ===
            if save_intermediate and header is not None:
                if recon_orientation == 'axial':
                    # 3a. source images
                    for i, source_img in enumerate(source_images):
                        img_save = source_img.squeeze().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
                        img_save = img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96]
                        nib.Nifti1Image(img_save, None, header).to_filename(
                            intermediate_out_dir / f'{prefix}_source{i}.nii.gz'
                        )
                    # 3b. beta images
                    beta = torch.stack(betas, dim=-1)
                    if len(beta.shape) > 4:
                        beta = beta.squeeze()
                    beta = beta.permute(1, 2, 0, 3).permute(1, 0, 2, 3).cpu().numpy()
                    img_save = nib.Nifti1Image(beta[112 - 96:112 + 96, :, 112 - 96:112 + 96, :], None, header)
                    file_name = intermediate_out_dir / f'{prefix}_source_betas.nii.gz'
                    nib.save(img_save, file_name)
                    # 3c. theta/eta values
                    with open(intermediate_out_dir / f'{prefix}_sources.txt', 'w') as fp:
                        fp.write(','.join(['img', 'slice'] + [f'theta{i}' for i in range(self.theta_dim)] +
                                          [f'eta{i}' for i in range(self.eta_dim)]) + '\n')
                        for i, img_key in enumerate([key.squeeze().cpu().numpy().tolist() for key in keys]):
                            for j, slice_key in enumerate(img_key):
                                fp.write(','.join([f'source{i}', f'slice{j:03d}'] +
                                                  ['%.6f' % val for val in slice_key]) + '\n')
                                
                else:
                    raise NotImplementedError('Only axial orientation is supported')

            # ===4. DECODING===
            for tid, (theta_target, query, norm_val) in enumerate(zip(thetas_target, queries, norm_vals)):
                if out_paths is not None:
                    out_prefix = out_paths[tid].name.replace('.nii.gz', '')
                rec_image, beta_fusion, logit_fusion, attention = [], [], [], []
                for batch_id in range(num_batches):
                    keys_tmp = [divide_into_batches(ks, num_batches)[batch_id] for ks in keys]
                    logits_tmp = [divide_into_batches(ls, num_batches)[batch_id] for ls in logits]
                    masks_tmp = [divide_into_batches(ms, num_batches)[batch_id] for ms in masks]
                    batch_size = keys_tmp[0].shape[0]
                    query_tmp = query.view(1, self.theta_dim + self.eta_dim, 1).repeat(batch_size, 1, 1)
                    k = torch.cat(keys_tmp, dim=-1).view(batch_size, self.theta_dim + self.eta_dim, 1, len(source_images))
                    v = torch.stack(logits_tmp, dim=-1).view(batch_size, self.beta_dim, 224 * 224, len(source_images))
                    logit_fusion_tmp, attention_tmp = self.attention_module(query_tmp, k, v, None, 5.0)
                    beta_fusion_tmp = self.channel_aggregation(reparameterize_logit(logit_fusion_tmp))
                    combined_map = torch.cat([beta_fusion_tmp, theta_target.repeat(batch_size, 1, 224, 224)], dim=1)
                    rec_image_tmp = self.decoder(combined_map) * masks_tmp[0]

                    rec_image.append(rec_image_tmp)
                    beta_fusion.append(beta_fusion_tmp)
                    logit_fusion.append(logit_fusion_tmp)
                    attention.append(attention_tmp)

                rec_image = torch.cat(rec_image, dim=0)
                beta_fusion = torch.cat(beta_fusion, dim=0)
                logit_fusion = torch.cat(logit_fusion, dim=0)
                attention = torch.cat(attention, dim=0)

                # ===5. SAVE INTERMEDIATE RESULTS (IF REQUESTED)===
                # harmonized image
                if header is not None:
                    if recon_orientation == "axial":
                        img_save = np.array(rec_image.cpu().squeeze().permute(1, 2, 0).permute(1, 0, 2))
                    else:
                        raise NotImplementedError('Only axial orientation is supported')
                    img_save = nib.Nifti1Image((img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96]) * norm_val, None,
                                               header)
                    file_name = out_path.parent / f'{out_prefix}_harmonized_{recon_orientation}.nii.gz'
                    nib.save(img_save, file_name)

                if save_intermediate and header is not None:
                    
                    if recon_orientation == 'axial':
                        # 5a. beta fusion
                        img_save = beta_fusion.squeeze().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_beta_fusion.nii.gz'
                        nib.save(img_save, file_name)

                        # 5b. logit fusion
                        img_save = logit_fusion.permute(2, 3, 0, 1).permute(1, 0, 2, 3).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96, :], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_logit_fusion.nii.gz'
                        nib.save(img_save, file_name)
                        
                        # 5c. attention
                        img_save = attention.permute(2, 3, 0, 1).permute(1, 0, 2, 3).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_attention.nii.gz'
                        nib.save(img_save, file_name)
                    else:
                        raise NotImplementedError('Only axial orientation is supported')

        if header is None:
            return rec_image.cpu().squeeze()
