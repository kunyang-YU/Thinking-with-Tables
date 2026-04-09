# MultimodalQA 数据路径说明

## 总览

```text
data/multimodalQA-used/
├─ adoption/
│  ├─ images/
│  │  ├─ train/{class}/*.jpg
│  │  └─ test/{class}/*.jpg
│  ├─ train.csv (表格数据)
│  └─ test.csv
├─ pawpularity/
│  ├─ dataset/
│  │  ├─ images/*.jpg
│  │  ├─ train.txt (回归任务数据标签标注)
│  │  └─ test.txt
│  ├─ train.csv
│  └─ test.csv
├─ paintings/
│  ├─ dataset/
│  │  ├─ images/*.jpg
│  │  ├─ train.txt
│  │  └─ test.txt
│  ├─ train.csv
│  └─ test.csv
└─ skinca/
   ├─ dataset/
   │  ├─ train/{class}/*.png
   │  └─ test/{class}/*.png
   ├─ train.csv
   └─ test.csv
```

---

## 1) Adoption

- **图片目录**
  - `data/multimodalQA-used/adoption/images/train/<class_id>/*.jpg`
  - `data/multimodalQA-used/adoption/images/test/<class_id>/*.jpg`
- **标签文件**
  - `data/multimodalQA-used/adoption/train.csv`
  - `data/multimodalQA-used/adoption/test.csv`
- **标签列**
  - `AdoptionSpeed`（分类标签）
- **图像键**
  - `PetID`（通常与图片名 `PetID-*.jpg` 对齐）

---

## 2) Pawpularity

- **图片目录**
  - `data/multimodalQA-used/pawpularity/dataset/images/*.jpg`
- **标签文件**
  - `data/multimodalQA-used/pawpularity/train.csv`
  - `data/multimodalQA-used/pawpularity/test.csv`
- **标签列**
  - `Pawpularity`（回归标签）
- **图像键**
  - `Id`（通常对应图片名 `Id.jpg`）
- **补充（已提供图文对齐列表）**
  - `data/multimodalQA-used/pawpularity/dataset/train.txt`
  - `data/multimodalQA-used/pawpularity/dataset/test.txt`
  - 每行格式：`<absolute_image_path>  <label>`

---

## 3) Paintings

- **图片目录**
  - `data/multimodalQA-used/paintings/dataset/images/*.jpg`
- **标签文件**
  - `data/multimodalQA-used/paintings/train.csv`
  - `data/multimodalQA-used/paintings/test.csv`
- **标签列**
  - `price`（回归标签）
- **图像键**
  - `image_url`（CSV 文件名对应 `dataset/images/` 下图片）
- **补充（已提供图文对齐列表）**
  - `data/multimodalQA-used/paintings/dataset/train.txt`
  - `data/multimodalQA-used/paintings/dataset/test.txt`
  - 每行格式：`<absolute_image_path>  <label>`

---

## 4) SkinCA

- **图片目录**
  - 按类划分图：`data/multimodalQA-used/skinca/dataset/train/<class_id>/*.png` 与 `.../test/<class_id>/*.png`
- **标签文件**
  - `data/multimodalQA-used/skinca/train.csv`
  - `data/multimodalQA-used/skinca/test.csv`
- **标签列**
  - `biopsed`（二分类）
  - `diagnostic`（多分类）
- **图像键**
  - `img_id`（CSV 中图像文件名）

---