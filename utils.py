#%% import library yang digunakan
import torch 
import torch.nn as nn 
import torch.optim as optim 

from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2 

import os 
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import matplotlib.patches as patches 

#%%
# Mendefinisikan fungsi untuk menghitung Intersection over Union (IoU) 
def iou(box1, box2, is_pred=True): 
    if is_pred: 
        # IoU score untuk prediksi dan label
        # box1 merupakan prediksi, box2 merupakan label
        # Format [x, y, lebar, tinggi].
		
        # Menghitung koordinat bounding box prediksi
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Koordinat bounding box ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Mendapatkan koordinat persegi panjang yang tumpang tindih
        x1 = torch.max(b1_x1, b2_x1) 
        y1 = torch.max(b1_y1, b2_y1) 
        x2 = torch.min(b1_x2, b2_x2) 
        y2 = torch.min(b1_y2, b2_y2) 
        # Memastikan bahwa tumpang tindih minimal 0 
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 

        # Menghitung luas gabungan
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) 
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) 
        union = box1_area + box2_area - intersection 

        # Menghitung skor IoU
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon) 

        # Mengembalikan skor IoU 
        return iou_score 
	
    else: 
        # Skor IoU berdasarkan lebar dan tinggi bounding box 
        
        # Menghitung luas tumpang tindih
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1]) 

        # Menghitung luas gabungan
        box1_area = box1[..., 0] * box1[..., 1] 
        box2_area = box2[..., 0] * box2[..., 1] 
        union_area = box1_area + box2_area - intersection_area 

        # Menghitung skor IoU
        iou_score = intersection_area / union_area 

        # Mengembalikan skor IoU 
        return iou_score

#%%
# Fungsi non-maximum suppression untuk menghapus bounding boxes yang tumpang tindih
def nms(bboxes, iou_threshold, threshold): 
    # Menyaring bounding boxes dengan kepercayaan di bawah ambang batas.
    bboxes = [box for box in bboxes if box[1] > threshold] 

    # Mengurutkan bounding boxes berdasarkan kepercayaan secara menurun.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 

    # Inisialisasi daftar bounding boxes setelah non-maximum suppression.
    bboxes_nms = [] 

    while bboxes: 
        # Mengambil bounding box pertama.
        first_box = bboxes.pop(0) 

        # Melakukan iterasi pada bounding boxes yang tersisa.
        for box in bboxes: 
            # Jika bounding boxes tidak tumpang tindih atau jika bounding box pertama
            # memiliki kepercayaan lebih tinggi, maka tambahkan bounding box kedua
            # ke dalam daftar bounding boxes setelah non-maximum suppression.
            if box[0] != first_box[0] or iou( 
                torch.tensor(first_box[2:]), 
                torch.tensor(box[2:]), 
            ) < iou_threshold: 
                # Periksa apakah box belum ada di bboxes_nms
                if box not in bboxes_nms: 
                    # Tambahkan box ke bboxes_nms 
                    bboxes_nms.append(box) 

    # Mengembalikan bounding boxes setelah non-maximum suppression.
    return bboxes_nms

#%%
# Fungsi untuk mengonversi sel menjadi bounding boxes 
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True): 
    # Batch size yang digunakan pada prediksi 
    batch_size = predictions.shape[0] 
    # Jumlah anchors 
    num_anchors = len(anchors) 
    # Daftar semua prediksi bounding boxes 
    box_predictions = predictions[..., 1:5] 

    # Jika input adalah prediksi, kita akan melewati koordinat x dan y
    # melalui fungsi sigmoid dan lebar serta tinggi melalui fungsi eksponensial,
    # kemudian menghitung skor dan kelas terbaik.
    if is_predictions: 
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) 
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) 
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors 
        scores = torch.sigmoid(predictions[..., 0:1]) 
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) 
    
    # Jika tidak, kita hanya akan menghitung skor dan kelas terbaik.
    else: 
        scores = predictions[..., 0:1] 
        best_class = predictions[..., 5:6] 

    # Menghitung indeks sel
    cell_indices = ( 
        torch.arange(s) 
        .repeat(predictions.shape[0], 3, s, 1) 
        .unsqueeze(-1) 
        .to(predictions.device) 
    ) 

    # Menghitung x, y, lebar, dan tinggi dengan penskalaan yang benar
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices) 
    y = 1 / s * (box_predictions[..., 1:2] +
                cell_indices.permute(0, 1, 3, 2, 4)) 
    width_height = 1 / s * box_predictions[..., 2:4] 

    # Menggabungkan nilai-nilai tersebut dan mengubah bentuknya menjadi
    # (BATCH_SIZE, num_anchors * S * S, 6) 
    converted_bboxes = torch.cat( 
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6) 

    # return daftar bounding boxes yang sudah diubah bentuk
    return converted_bboxes.tolist()


#%%
# Fungsi untuk menampilkan gambar dengan bounding boxes dan label kelas 
def plot_image(image, boxes): 
    # Mengambil peta warna dari matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    # Mendapatkan 20 warna berbeda dari peta warna untuk 72 kelas yang berbeda 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 

    # Membaca gambar dengan OpenCV 
    img = np.array(image) 
    # Mendapatkan tinggi dan lebar gambar 
    h, w, _ = img.shape 

    # Membuat objek figure dan axes 
    fig, ax = plt.subplots(1) 

    # Menambahkan gambar ke plot 
    ax.imshow(img) 

    # Memplot bounding boxes dan label di atas gambar 
    for box in boxes: 
        # Mendapatkan kelas dari bounding box 
        class_pred = box[0] 
        # Mendapatkan koordinat pusat x dan y 
        box = box[2:] 
        # Mendapatkan koordinat sudut kiri atas 
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        # Membuat objek Rectangle patch dengan bounding box 
        rect = patches.Rectangle( 
            (upper_left_x * w, upper_left_y * h), 
            box[2] * w, 
            box[3] * h, 
            linewidth=2, 
            edgecolor=colors[int(class_pred)], 
            facecolor="none", 
        ) 
        
        # Menambahkan patch ke Axes 
        ax.add_patch(rect) 
        
        # Menambahkan nama kelas ke patch 
        plt.text( 
            upper_left_x * w, 
            upper_left_y * h, 
            s=class_labels[int(class_pred)], 
            color="white", 
            verticalalignment="top", 
            bbox={"color": colors[int(class_pred)], "pad": 0}, 
        ) 

    # Menampilkan plot 
    plt.show()

#%%
# Function untuk save checkpoint 
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"): 
	print("==> Saving checkpoint") 
	checkpoint = { 
		"state_dict": model.state_dict(), 
		"optimizer": optimizer.state_dict(), 
	} 
	torch.save(checkpoint, filename)
#%%
# Function untuk load checkpoint 
def load_checkpoint(checkpoint_file, model, optimizer, lr): 
	print("==> Loading checkpoint") 
	checkpoint = torch.load(checkpoint_file, map_location=device) 
	model.load_state_dict(checkpoint["state_dict"]) 
	optimizer.load_state_dict(checkpoint["optimizer"]) 

	for param_group in optimizer.param_groups: 
		param_group["lr"] = lr 
#%%
# Perangkat (device)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Variabel untuk memuat dan menyimpan model
load_model = False  # Jika True, model akan dimuat
save_model = True   # Jika True, model akan disimpan
checkpoint_file = "checkpoint.pth.tar"  # Nama file untuk checkpoint model

# Kotak anchor untuk setiap peta fitur, yang berskala antara 0 dan 1
# Terdapat 3 peta fitur pada 3 skala yang berbeda berdasarkan paper YOLOv3
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

# Batch size untuk pelatihan
batch_size = 32

# Learning rate untuk pelatihan
learning_rate = 1e-5

# Jumlah epoch untuk pelatihan
epochs = 20

# Ukuran gambar
image_size = 416

# Ukuran grid cell
s = [image_size // 32, image_size // 16, image_size // 8]

# Label kelas
class_labels = [
    'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]

#%%
# Kelas dataset untuk memuat gambar dan label dari folder
class Dataset(torch.utils.data.Dataset): 
    def __init__( 
        self, csv_file, image_dir, label_dir, anchors, 
        image_size=416, grid_sizes=[13, 26, 52], 
        num_classes=20, transform=None
    ): 
        # Membaca file csv dengan nama gambar dan label
        self.label_list = pd.read_csv(csv_file) 
        # Direktori gambar dan label
        self.image_dir = image_dir 
        self.label_dir = label_dir 
        # Ukuran gambar
        self.image_size = image_size 
        # Transformasi
        self.transform = transform 
        # Ukuran grid untuk setiap skala
        self.grid_sizes = grid_sizes 
        # Kotak anchor
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) 
        # Jumlah kotak anchor
        self.num_anchors = self.anchors.shape[0] 
        # Jumlah kotak anchor per skala
        self.num_anchors_per_scale = self.num_anchors // 3
        # Jumlah kelas
        self.num_classes = num_classes 
        # Ambang IoU untuk diabaikan
        self.ignore_iou_thresh = 0.5

    def __len__(self): 
        return len(self.label_list) 
    
    def __getitem__(self, idx): 
        # Mendapatkan path label
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1]) 
        # Mengaplikasikan roll untuk memindahkan label kelas ke kolom terakhir
        # 5 kolom: x, y, lebar, tinggi, label_kelas 
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() 
        
        # Mendapatkan path gambar
        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0]) 
        image = np.array(Image.open(img_path).convert("RGB")) 

        # Augmentasi menggunakan Albumentations
        if self.transform: 
            augs = self.transform(image=image, bboxes=bboxes) 
            image = augs["image"] 
            bboxes = augs["bboxes"] 

        # Mendeklarasikan target dengan asumsi 3 prediksi skala (seperti pada paper) 
        # dan jumlah kotak anchor yang sama per skala
        # target : [probabilitas, x, y, lebar, tinggi, label_kelas] 
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) for s in self.grid_sizes] 
        
        # Mengidentifikasi kotak anchor dan sel untuk setiap bounding box 
        for box in bboxes: 
            # Menghitung IoU dari bounding box dengan kotak anchor 
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False) 
            # Memilih kotak anchor terbaik 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
            x, y, width, height, class_label = box 

            # Pada setiap skala, mengassign bounding box ke kotak anchor yang cocok terbaik 
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices: 
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
                
                # Mengidentifikasi ukuran grid untuk skala tersebut 
                s = self.grid_sizes[scale_idx] 
                
                # Mengidentifikasi sel tempat bounding box berada 
                i, j = int(s * y), int(s * x) 
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
                
                # Memeriksa apakah kotak anchor sudah diassign 
                if not anchor_taken and not has_anchor[scale_idx]: 
                    # Mengatur probabilitas menjadi 1 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Menghitung pusat bounding box relatif terhadap sel 
                    x_cell, y_cell = s * x - j, s * y - i 

                    # Menghitung lebar dan tinggi bounding box relatif terhadap sel 
                    width_cell, height_cell = (width * s, height * s) 

                    # Mengidentifikasi koordinat kotak 
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) 

                    # Mengassign koordinat kotak ke target 
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 

                    # Mengassign label kelas ke target 
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 

                    # Mengatur kotak anchor sebagai sudah diassign untuk skala tersebut 
                    has_anchor[scale_idx] = True

                # Jika kotak anchor sudah diassign, periksa apakah IoU lebih besar dari ambang batas 
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
                    # Mengatur probabilitas menjadi -1 untuk mengabaikan kotak anchor 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Mengembalikan gambar dan target 
        return image, tuple(targets)
#%%
# Transformasi untuk pelatihan
train_transform = A.Compose( 
    [ 
        # Merescale gambar sehingga sisi maksimum sama dengan image_size
        A.LongestMaxSize(max_size=image_size), 
        # Menambahkan padding pada area yang tersisa dengan nilai nol
        A.PadIfNeeded( 
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Jittering warna secara acak
        A.ColorJitter( 
            brightness=0.5, contrast=0.5, 
            saturation=0.5, hue=0.5, p=0.5
        ), 
        # Memutar gambar secara horizontal
        A.HorizontalFlip(p=0.5), 
        # Normalisasi gambar
        A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ), 
        # Mengubah gambar menjadi tensor PyTorch
        ToTensorV2() 
    ], 
    # Augmentasi untuk bounding boxes 
    bbox_params=A.BboxParams( 
        format="yolo", 
        min_visibility=0.4, 
        label_fields=[] 
    ) 
) 

# Transformasi untuk pengujian
test_transform = A.Compose( 
    [ 
        # Merescale gambar sehingga sisi maksimum sama dengan image_size
        A.LongestMaxSize(max_size=image_size), 
        # Menambahkan padding pada area yang tersisa dengan nilai nol
        A.PadIfNeeded( 
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Normalisasi gambar
        A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ), 
        # Mengubah gambar menjadi tensor PyTorch
        ToTensorV2() 
    ], 
    # Augmentasi untuk bounding boxes 
    bbox_params=A.BboxParams( 
        format="yolo", 
        min_visibility=0.4, 
        label_fields=[] 
    ) 
)
#%%
# Mendapatkan gambar
dataset = Dataset( 
    csv_file="train.csv", 
    image_dir="images/", 
    label_dir="labels/", 
    grid_sizes=[13, 26, 52], 
    anchors=ANCHORS, 
    transform=test_transform 
) 

# Membuat objek dataloader 
loader = torch.utils.data.DataLoader( 
    dataset=dataset, 
    batch_size=1,   
    shuffle=True, 
) 

# Mendefinisikan ukuran grid dan anchor yang diskalakan 
GRID_SIZE = [13, 26, 52] 
scaled_anchors = torch.tensor(ANCHORS) / ( 
    1 / torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
) 

# Mendapatkan batch dari dataloader 
x, y = next(iter(loader)) 

# Mendapatkan koordinat kotak dari label 
# dan mengonversinya menjadi bounding box tanpa penskalaan 
boxes = [] 
for i in range(y[0].shape[1]): 
    anchor = scaled_anchors[i] 
    boxes += convert_cells_to_bboxes( 
            y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor 
            )[0] 

# Mengaplikasikan non-maximum suppression 
boxes = nms(boxes, iou_threshold=1, threshold=0.7) 

# Menampilkan gambar dengan bounding box 
plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)

#%%
