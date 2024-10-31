
import numpy as np
import torch
import os

from datetime import datetime
import nibabel as nib

from utils import reparameterize_logit, divide_into_batches

from network import UNet, ThetaEncoder, EtaEncoder, Patchifier, AttentionModule


class HAIL:
    """
    Harmonization Across Imaging Locations (HAIL) model.
    """
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

    def channel_aggregation(self, beta_onehot_encode: torch.Tensor) -> torch.Tensor:
        """
        Combine multi-channel one-hot encoded beta into one channel (label-encoding).

        args:
            beta_onehot_encode: torch.Tensor (batch_size, self.beta_dim, image_dim, image_dim)
                One-hot encoded beta variable. At each pixel location, only one channel will take value of 1,
                and other channels will be 0.
        return: 


        """
        
        batch_size, image_dim = beta_onehot_encode.shape[0], beta_onehot_encode.shape[3]
        value_tensor = (torch.arange(0, self.beta_dim) * 1.0).to(self.device)
        value_tensor = value_tensor.view(1, self.beta_dim, 1, 1).repeat(batch_size, 1, image_dim, image_dim)
        beta_label_encode = beta_onehot_encode * value_tensor.detach()
        return beta_label_encode.sum(1, keepdim=True) / self.beta_dim


    def harmonize(self, source_images, target_images, target_theta, target_eta, out_paths,
                  recon_orientation, norm_vals, header=None, num_batches=4) -> [torch.Tensor | None]:
        """
         The main hamronization function that harmonizes the source images to the target images.

        Args:
            source_images (List[torch.Tensor]): list of source images
            target_images (List[torch.Tensor]): list of target images
            target_theta (List[torch.Tensor]): list of target theta values
            target_eta (List[torch.Tensor]): list of target eta values
            out_paths (List[Path]): list of output paths
            recon_orientation (str): orientation of the reconstructed image
            norm_vals (List[Tuple[int]]): list of normalization values for the reconstructed image
            header (nib.Nifti1Header): header of the input image
            num_batches (int): number of batches to divide the input tensor into

        Returns:
            torch.Tensor: reconstructed harmonized image
        """
        if out_paths is not None:
            for out_path in out_paths:
                os.makedirs(out_path.parent, exist_ok=True)
            prefix = str(out_paths[0].name).split('.')[0]
        
        # set everything to an eval mode and turn off the gradient
        with torch.set_grad_enabled(False):
            self.beta_encoder.eval()
            self.theta_encoder.eval()
            self.eta_encoder.eval()
            self.decoder.eval()

            # Calculate the masks, logits, betas, and keys for the source images
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

            # calculate the harmonized theta and eta from the target images
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
            else:
                queries, thetas_target = [], []
                for target_theta_tmp, target_eta_tmp in zip(target_theta, target_eta):
                    thetas_target.append(target_theta_tmp.view(1, self.theta_dim, 1, 1).to(self.device))
                    queries.append(torch.cat([target_theta_tmp.view(1, self.theta_dim, 1).to(self.device),
                                              target_eta_tmp.view(1, self.eta_dim, 1).to(self.device)], dim=1))

            # decode the harmonized normal val  image
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

                # save the harmonized image
                print('recon_orient', recon_orientation)
                if header is not None:
                    if recon_orientation == "axial":
                        img_save = np.array(rec_image.cpu().squeeze().permute(1, 2, 0).permute(1, 0, 2))
                    else:
                        raise NotImplementedError('Only axial orientation is supported')
                    
                    # put the image back to the original shape
                    # put the image back to original harmonized intensity
                    img_save = self.clipper(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96]* norm_val[0], norm_val[1])
                    img_save = nib.Nifti1Image(img_save, None,
                                               header)
                    file_name = out_path.parent / f'{out_prefix}_harmonized_{recon_orientation}.nii.gz'
                    nib.save(img_save, file_name)


        if header is None:
            return rec_image.cpu().squeeze()

    def clipper(self, rec_image: np.ndarray, norm_val: float) -> np.ndarray:
        """ Clip the image to the valid intensity

        Args:
            rec_image (np.ndarray): harmonized image
            norm_val (float ): normalization value to be clipped to

        Returns:
            np.ndarray : clipped image
        """
        return  np.clip(rec_image, 0, norm_val)
