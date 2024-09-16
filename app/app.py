import logging
import subprocess
import glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import PIL
import gradio as gr
import os
from app_assets import logo

logger = logging.getLogger(__file__)
# Constants

# Always use dumy name to maintain annonimity 
DUMMY_DIR = "HARMPATH"
DUMMY_FILE_NAMES = {"t1" : "t1.nii.gz" , "template" : "template.nii.gz"}


mydict = {}

def run_inference(image_t1, image_template):

    input_path = Path(f'/tmp/peds-app/{DUMMY_DIR}')
    os.makedirs(input_path, exist_ok=True)
    output_folder = Path('./outs')
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder / f'harm_{Path(image_t1).name}'
    fake_output_path = output_folder / f'{DUMMY_DIR}.nii.gz'

    subprocess.run(f"cp {image_t1} {output_path}", shell=True)

    return image_t1, image_template, output_path

def get_img(image_path):
    """_summary_

    Args:
        image_path (_type_): _description_
        mask_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    img_obj = sitk.ReadImage(image_path)
    img = sitk.GetArrayFromImage(img_obj)
    
    # Normalize image
    minval, maxval = np.min(img), np.max(img)
    img = ((img - minval) / (maxval - minval)).clip(0, 1) * 255


    return img

def render_slice(img, x, view):
    if view == 'axial':
        slice_img = img[x, :, :] 
    elif view == 'coronal':
        slice_img = img[:, x, :] 
    elif view == 'sagittal':
        slice_img  = img[:, :, x] 
    
    slice_img = np.flipud(slice_img)
    return slice_img

def main_func(image_t1, image_template,  file_to_render):
    print(image_t1, image_template,  file_to_render)
    global mydict
    image_path, template_path, harm_path = run_inference(Path(image_t1), Path(image_template))
    mydict['img_path'] = image_path
    mydict['temp_path'] = template_path
    mydict['harm_path'] = harm_path
    return str(harm_path)


def render(file_to_render, x, view):
    if 'img_path' in mydict:
        img = get_img(mydict[file_to_render])
        max_render= img.shape[0 if view == 'axial' else (1 if view == 'coronal' else 2)] - 1
        print( max_render)
        x = max(0, min(x, max_render))
        slice_img= render_slice(img, x, view)
        
        im = PIL.Image.fromarray(slice_img.astype(np.uint8))
        annotations = [
            (np.zeros_like(slice_img), f""),
        ]

        return im, annotations
    else:
        return np.zeros((10, 10)), []

def render_axial(file_to_render, x):
    return render('img_path', x, 'axial')
def render_coronal(file_to_render, x):
    return render('img_path', x, 'coronal')
def render_sagittal(file_to_render, x):
    return render('img_path', x, 'sagittal')

def render_axial_template(file_to_render, x):
    return render('temp_path', x, 'axial')
def render_coronal_template(file_to_render, x):
    return render('temp_path', x, 'coronal')
def render_sagittal_template(file_to_render, x):
    return render('temp_path', x, 'sagittal')
# Gradio UI
with gr.Blocks() as demo:

    gr.HTML(value=f"<center><font size='6'><bold> Harmonization Across Imaging Locations (HAIL)</bold></font></center>")
    gr.HTML(value=f"<p style='margin-top: 1rem, margin-bottom: 1rem'> <img src='{logo.logo}' alt='Childrens National Logo' style='display: inline-block'/></p>")
    gr.HTML(value=f"<justify><font size='4'> Welcome to the pediatric brain tumor segmenter. Please read the <a href='https://precision-medical-imaging-group.github.io/SegmenterApp-HAIL/'>instructions</a> before using the application. </font></justify>")
    with gr.Row():
 
        image_t1 = gr.File(label="upload t1 image to be harmonized:", file_types=["nii.gz"])
        image_template = gr.File(label="upload t1 template image:", file_types=["nii.gz"])

    with gr.Row():
        with gr.Column():
            render_t1 = gr.Dropdown(['native T1', 'post-contrast T1-weighted'], label='choose the type of t1 image uploaded', render=False)
        with gr.Column():
            file_to_render = gr.Dropdown(['native T1', 'post-contrast T1-weighted'], label='choose the type of t1 image uploaded')
        with gr.Column():
            render_temp = gr.Dropdown(['native T1', 'post-contrast T1-weighted'], label='choose the type of t1 image uploaded', render=False)

    with gr.Row():
        with gr.Column():
             gr.Button("", render=False)
        with gr.Column():
            btn = gr.Button("Start Harmonization")
        with gr.Column():
             gr.Button("", render=False)
    
    gr.HTML(value=f"<justify><font size='4'> INPUT NifTi: </font></justify>")
    with gr.Row():
        height = "20vw"
        myimage_axial = gr.AnnotatedImage(label="axial view", height=height)
        myimage_coronal = gr.AnnotatedImage(label="coronal view",height=height)
        myimage_sagittal = gr.AnnotatedImage(label="sagittal view",height=height)
    gr.HTML(value=f"<justify><font size='4'> HARMONIZED NifTi: </font></justify>")
    with gr.Row():
        height = "20vw"
        output_axial_template = gr.AnnotatedImage(label="axial view", height=height)
        output_coronal_template = gr.AnnotatedImage(label="coronal view",height=height)
        output_sagittal_template = gr.AnnotatedImage(label="sagittal view",height=height)

    with gr.Row():
        slider_axial = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_axial = gr.State(value=75)
        slider_coronal = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_coronal = gr.State(value=75)
        slider_sagittal = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_sagittal = gr.State(value=75)

    with gr.Row():
        mask_file = gr.File(label="Download Harmonized File", height="vw" )

    # example_dir = '/home/pmilab/Abhijeet/examples/'
    # generate_examples = glob.glob(example_dir + '*')
    # order_list = ['-t1c.nii.gz', '-t2f.nii.gz', '-t1n.nii.gz', '-t2w.nii.gz']
    # example_list = [[os.path.join(path, str(Path(path).name)+ending) for ending in order_list] for path in generate_examples]

    
    # # gr.HTML(value=f"<center><font size='2'> The software is provided 'as is', without any warranties or liabilities.  For research use only and not intended for medical diagnosis. We do not store or access any information uploaded to the platform. This version is v20240827.</font></center>")
    # gr.Examples(
    #     examples=example_list,
    #     inputs=[image_t1c, image_t2f, image_t1n, image_t2w],
    #     outputs=[mask_file,out_text],
    #     fn=main_func,
    #     cache_examples=False,
    #     label="Preloaded BraTS 2023 examples"
    # )
    
    btn.click(fn=main_func, 
        inputs=[image_t1, image_template,  file_to_render], outputs=[mask_file],
    )
    file_to_render.select(render_axial,
        inputs=[file_to_render, state_axial],
        outputs=[output_axial_template])
    file_to_render.select(render_axial,
        inputs=[file_to_render, state_axial],
        outputs=[myimage_axial])


    file_to_render.select(render_coronal,
        inputs=[file_to_render, state_coronal],
        outputs=[output_coronal_template])
    file_to_render.select(render_coronal,
        inputs=[file_to_render, state_coronal],
        outputs=[myimage_coronal])

    file_to_render.select(render_sagittal,
        inputs=[file_to_render, state_sagittal],
        outputs=[output_sagittal_template])
    file_to_render.select(render_sagittal,
        inputs=[file_to_render, state_sagittal],
        outputs=[myimage_sagittal])


    slider_axial.change(
        render_axial,
        inputs=[file_to_render, slider_axial],
        outputs=[myimage_axial],
        api_name="axial_slider"
    )
    slider_axial.change(
        render_axial_template,
        inputs=[file_to_render, slider_axial],
        outputs=[output_axial_template],
        api_name="axial_slider"
    )

    slider_coronal.change(
        render_coronal,
        inputs=[file_to_render, slider_coronal],
        outputs=[myimage_coronal],
        api_name="axial_slider"
    )
    slider_coronal.change(
        render_coronal_template,
        inputs=[file_to_render, slider_coronal],
        outputs=[output_coronal_template],
        api_name="axial_slider"
    )

    slider_sagittal.change(
        render_sagittal,
        inputs=[file_to_render, slider_sagittal],
        outputs=[myimage_sagittal],
        api_name="axial_slider"
    )
    slider_sagittal.change(
        render_sagittal_template,
        inputs=[file_to_render, slider_sagittal],
        outputs=[output_sagittal_template],
        api_name="axial_slider"
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0")