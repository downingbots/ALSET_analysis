from STEGO.src.modules import *
# import hydra
from collections import defaultdict
import cv2
import numpy
import torch.multiprocessing
from PIL import Image
from STEGO.src.crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from STEGO.src.train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import random
torch.multiprocessing.set_sharing_strategy('file_system')


class UnlabeledImageFolder(Dataset):
    # def __init__(self, root, transform):
    def __init__(self, img, transform):
        super(UnlabeledImageFolder, self).__init__()
        # self.root = join(root)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        self.root = img_pil
        self.transform = transform
        # self.images = os.listdir(self.root)
        self.images = img_pil

    def __getitem__(self, index):
        # image = Image.open(join(self.root, self.images[index])).convert('RGB')
        image = self.images
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        return image, "image"
        # return image, self.images[index]

    def __len__(self):
        # return len(self.images)
        return 1

# @hydra.main(config_path="configs", config_name="demo_config.yml")
# def my_app(cfg: DictConfig) -> None:
def stego(img, unique_color):

    cfg = OmegaConf.load("alset_stego_config.yml")
    result_dir = "./STEGO/results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "linear"), exist_ok=True)

    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.model_path)
    # print(OmegaConf.to_yaml(model.cfg))

    dataset = UnlabeledImageFolder(
         # root=cfg.image_dir,
         img=img,
         transform=get_transform(cfg.res, False, "center"),
    )

    # loader = DataLoader(dataset, cfg.batch_size * 2,
    #                   shuffle=False, num_workers=cfg.num_workers,
    #                   pin_memory=True, collate_fn=flexible_collate)
    loader = DataLoader(dataset, 1,
                        shuffle=False, num_workers=2,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()
    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net


    # tqdm provides a progress bar
    #
    # loader identifies the image’s location on disk, 
    # converts that to a tensor using read_image, 
    # retrieves the corresponding label from the csv data in self.img_labels,
    # calls the transform functions on them (if applicable),
    # and returns the tensor image and corresponding label in a tuple.
    #
    for i, (img, name) in enumerate(tqdm(loader)):
    # if True:
        with torch.no_grad():
            img = img.cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

            fig, ax = plt.subplots(3, 1, figsize=(1 * 3, 3 * 3))
            for j in range(img.shape[0]):
                single_img = img[j].cpu()
                linear_crf = dense_crf(single_img, linear_probs[j]).argmax(0)
                cluster_crf = dense_crf(single_img, cluster_probs[j]).argmax(0)
                print("linear, clust", linear_crf.shape, cluster_crf.shape)

                # new_name = ".".join(name[j].split(".")[:-1]) + ".png"
                new_name = "image.png"
                Image.fromarray(linear_crf.astype(np.uint8)).save(join(result_dir, "linear", new_name))
                Image.fromarray(cluster_crf.astype(np.uint8)).save(join(result_dir, "cluster", new_name))

                ############################
                saved_data = defaultdict(list)
                saved_data["img"].append(img.cpu())

                # plot_img = ((saved_data["img"][0]) * 255).unsqueeze(0).numpy().astype(np.uint8)
                plot_img = (cluster_crf*255).astype(np.uint8)
                cv_plot_img = np.zeros((len(plot_img), len(plot_img[0]), 3), dtype="uint8")
                num_color = 0
                for i,row in enumerate(plot_img):
                  for j,pix in enumerate(row):
                    try:
                      cv_plot_img[i][j] = unique_color[pix]
                    except:
                      unique_color[pix] = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
                      cv_plot_img[i][j] = unique_color[pix]
                # list_color = list(dataset.unique_color)
                # num_color = len(list_color)
                # for i,row in enumerate(plot_img):
                  # for j,pix in enumerate(row):
                    # cv_plot_img[i][j] = dataset.unique_color[pix]
                    
                print("unique colors:", unique_color.keys())
                # Image.fromarray(cluster_crf.astype(np.uint8)).save(join(result_dir, "cluster", new_name))
                # print("plot_img", plot_img.shape)
                # plot_img = numpy.squeeze(plot_img)
                # print("plot_img0", plot_img.shape)
                Image.fromarray(plot_img).save(join(join(result_dir, "img", str(0) + ".jpg")))

                # cv_image = numpy.array(Image.fromarray(plot_img).convert('RGB'))
                # cv_image = cv_image[:, :, ::-1].copy() # Convert RGB to BGR 

                # ax[0].imshow(plot_img)
                # ax[0].set_ylabel("Image", fontsize=26)
                # remove_axes(ax)
                # plt.tight_layout()
                # plt.show()
                return cv_plot_img, unique_color
                if cfg.run_prediction:
                    plot_cluster = (model.label_cmap[
                        model.test_cluster_metrics.map_clusters(
                            plot_img)]).astype(np.uint8)
                    # plot_cluster = (model.label_cmap[
                    #     model.test_cluster_metrics.map_clusters(
                    #         saved_data["cluster_preds"][img_num])]) \
                    #     .astype(np.uint8)
                    ax[1].imshow(plot_cluster)

            # ax[0].set_ylabel("Image", fontsize=26)
            # if cfg.run_prediction:
            #     ax[2].set_ylabel("STEGO\n(Ours)", fontsize=26)

            # remove_axes(ax)
            # plt.tight_layout()
            # plt.show()
            # plt.clf()



if __name__ == "__main__":
    unique_color = {}
    prep_args()
    my_app()
