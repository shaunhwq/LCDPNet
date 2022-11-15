import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import cv2
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim


def get_psnr_ssim(img1_path, img2_path):
    im1, im2 = cv2.imread(img1_path), cv2.imread(img2_path)
    psnr = calculate_psnr(im1, im2)
    ssim = calculate_ssim(im1, im2, multichannel=True)
    return psnr, ssim


if __name__ == "__main__":
    input_path1 = "checkpoints/test_result/lcdpnet_checkpoints_trained_on_ours.ckpt@ours-cslab.test"
    input_images = [os.path.join(input_path1, img) for img in os.listdir(input_path1)]
    print(input_images[0])
    input_images = [img for img in input_images if os.path.isfile(img) and os.path.basename(img)[0] != 0 and os.path.splitext(os.path.basename(img))[1] in {".jpg", ".JPG"}]
    input_images.sort()

    input_path2 = "/data2/shaun/datasets/exposure_correction/exposure_errors/testing/expert_c_testing_set"
    gt_images = [os.path.join(input_path2, img) for img in os.listdir(input_path2) if img[0] != "."] * 5
    gt_images.sort()


    with ProcessPoolExecutor() as executor:
        results = list(tqdm(iterable=executor.map(get_psnr_ssim, input_images, gt_images), total=len(input_images), desc="Getting PSNR/SSIM"))

    psnr_list = [result[0] for result in results]
    ssim_list = [result[1] for result in results]
    print(f"PSNR: {round(sum(psnr_list) / len(psnr_list), 3)}")
    print(f"SSIM: {round(sum(ssim_list) / len(ssim_list), 3)}")