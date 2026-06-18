import cv2
import streamlit as st

from rcnn_segmentation import (
    DEFAULT_MAX_IMAGE_DIMENSION,
    bgr_to_rgb,
    load_default_model,
    read_image_from_bytes,
    read_image_from_path,
    run_instance_segmentation,
)


def draw_bboxes_only(image, detections):
    """Gambar hanya bounding box (tanpa mask) dari deteksi yang lolos confidence filter."""
    img = image.copy()
    for d in detections:
        left, top, right, bottom = d["left"], d["top"], d["right"], d["bottom"]
        cv2.rectangle(img, (left, top), (right, bottom), (0, 180, 0), 2)
        label_text = f"{d['label']} {d['confidence']:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(top - 6, label_size[1] + 4)
        cv2.putText(img, label_text, (left, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 1)
    return img


def render_style():
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1100px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                padding-top: 2rem !important;
                padding-bottom: 2rem !important;
            }

            h1, h2, h3 {
                text-align: center;
            }

            /* Subtitle: explicitly white/dark depending on theme via forced readable color */
            .app-subtitle {
                text-align: center;
                color: #a0aec0;
                font-size: 1.05rem;
                margin-top: -0.5rem;
                margin-bottom: 1.5rem;
            }


            /* File uploader: full width, no broken layout */
            [data-testid="stFileUploader"] {
                width: 100% !important;
            }

            [data-testid="stFileUploader"] section {
                width: 100% !important;
                background-color: transparent !important;
                border: 1px dashed rgba(100, 116, 139, 0.5) !important;
                border-radius: 10px !important;
            }

            [data-testid="stFileUploader"] section > div {
                width: 100% !important;
            }

            [data-testid="stMetric"] {
                background-color: transparent !important;
                border: 1px solid rgba(100, 116, 139, 0.25) !important;
                border-radius: 10px !important;
                padding: 0.8rem 1rem !important;
            }

            /* Footer */
            .footer-custom {
                margin-top: 6rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(100, 116, 139, 0.2);
                text-align: center;
                color: #718096;
                font-size: 0.95rem;
            }

            /* Info badge label di samping uploader */
            .upload-label {
                font-size: 0.95rem;
                font-weight: 600;
                color: inherit;
                margin-bottom: 0.4rem;
            }
            /* Sembunyikan garis border kiri-kanan dalam expander */
            [data-testid="stExpander"] details {
                border-left: none !important;
                border-right: none !important;
                border-radius: 0 !important;
            }

            [data-testid="stExpander"] details > div {
                border-left: none !important;
                border-right: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def get_model():
    return load_default_model()

def get_input_image(uploaded_file):
    if uploaded_file is not None:
        return read_image_from_bytes(uploaded_file.getvalue()), uploaded_file.name

    return read_image_from_path("images/image_placeholder.png"), "images/image_placeholder.png"

def render_info_expander():
    with st.expander("Tentang Deep Instance Segmentation", expanded=False):
        st.markdown("""
        ### Apa itu Instance Segmentation?
        **Instance segmentation** adalah teknik computer vision yang memberikan label pada setiap piksel
        dalam gambar sehingga membentuk *mask* berbasis piksel untuk setiap objek secara individual.

        Perbedaannya dengan **semantic segmentation**:
        | Aspek | Semantic Segmentation | Instance Segmentation |
        |---|---|---|
        | Unit output | Per kelas | Per objek (instance) |
        | Contoh: 2 kucing + 1 anjing | 2 kelas unik | 3 objek unik |
        | Bisa bedakan individu? | Tidak | Ya |

        ---

        ### Mask R-CNN
        Model yang digunakan adalah **Mask R-CNN** (*Region-based CNN*), yang dibangun di atas Fast R-CNN
        dengan tambahan cabang prediksi *mask* piksel.

        **Alur kerja Mask R-CNN:**
        1. **Input**: Gambar dimasukkan ke jaringan (FPN + ResNet/Inception backbone)
        2. **Region Proposals**: Kandidat area objek dihasilkan oleh *Region Proposal Network* (RPN)
        3. **ROI Align**: Fitur diekstrak per proposal dengan presisi subpiksel
        4. **Head Classification**: Tiga output simultan - *class label*, *bounding box*, dan *mask*

        ---

        ### Model yang Digunakan
        - **Backbone**: Inception V2
        - **Dataset Training**: [COCO (Common Objects in Context)](http://cocodataset.org/) - **90 kelas**
        - **Format model**: TensorFlow frozen graph (`.pb`) + config OpenCV DNN (`.pbtxt`)
        - **Inferensi**: OpenCV `cv2.dnn` - tidak butuh instalasi TensorFlow/PyTorch

        ---

        ### OpenCV DNN
        **OpenCV DNN** (*Deep Neural Network module*) adalah modul bawaan OpenCV yang memungkinkan
        inferensi model deep learning tanpa perlu menginstal framework seperti TensorFlow atau PyTorch.

        Cara kerjanya dalam proyek ini:
        1. Model TensorFlow yang sudah dilatih disimpan sebagai *frozen graph* (`.pb`) - bobot model dibekukan
        2. OpenCV membaca file tersebut dengan `cv2.dnn.readNetFromTensorflow(weights, config)`
        3. Gambar dikonversi ke format *blob* dengan `cv2.dnn.blobFromImage`
        4. Forward pass dijalankan dengan `net.forward(["detection_out_final", "detection_masks"])`
        5. Output berupa koordinat bounding box dan mask segmentasi per kelas

        Keuntungan menggunakan OpenCV DNN: ringan, tidak bergantung GPU (bisa berjalan di CPU), dan terintegrasi langsung dengan pipeline pengolahan gambar OpenCV.

        ---

        ### Parameter
        - **Confidence threshold** (`conf = 0.5`): Deteksi dengan skor di bawah ini akan diabaikan
        - **Mask threshold** (`thresh = 0.3`): Piksel mask di bawah nilai ini dianggap bukan bagian objek
        - Warna mask di-*blend* 40% warna acak + 60% warna asli piksel gambar

        ---

        ### Referensi
        - [Girshick, 2014 - R-CNN](https://arxiv.org/pdf/1311.2524.pdf)
        - [Girshick, 2015 - Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
        - [Ren et al., 2015 - Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)
        - [He et al., 2017 - Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
        - [PyImageSearch - Mask R-CNN with OpenCV](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)
        """)

def main():
    st.set_page_config(
        page_title="Deep Instance Segmentation",
        layout="wide"
    )

    render_style()

    st.title("Deep Instance Segmentation")
    st.markdown(
        "<div class='app-subtitle'>Implementasi instance segmentation menggunakan pretrained Mask R-CNN dan OpenCV DNN.</div>",
        unsafe_allow_html=True
    )

    # Load model
    try:
        net, labels = get_model()
    except Exception as error:
        st.error("Model belum siap. Jalankan `python utils/download_models.py` terlebih dahulu.")
        st.code(str(error))
        return

    # File uploader - label di atas, bukan collapsed, agar tidak hilang dan tidak ke-enter
    st.markdown("<div class='upload-label'>Upload Gambar (PNG, JPG, JPEG, BMP, WEBP)</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload gambar untuk segmentasi",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed"
    )

    # Load image
    try:
        image, image_source = get_input_image(uploaded_file)
    except Exception as error:
        st.error("Gambar tidak bisa dibaca.")
        st.code(str(error))
        return

    # Run segmentation - dengan progress step-by-step
    status_placeholder = st.empty()

    with st.spinner(""):
        status_placeholder.info("Langkah 1/3 - Menyiapkan gambar (preprocessing blob)...")
        result = None

        try:
            # Jalankan segmentasi (preprocessing + forward pass + postprocessing)
            status_placeholder.info("Langkah 2/3 - Menjalankan forward pass Mask R-CNN...")
            result = run_instance_segmentation(
                image=image,
                net=net,
                labels=labels,
                conf=0.5,
                thresh=0.3,
                max_dimension=DEFAULT_MAX_IMAGE_DIMENSION
            )
            status_placeholder.info("Langkah 3/3 - Menggambar mask dan bounding box...")
        except Exception as error:
            status_placeholder.empty()
            st.error("Terjadi kesalahan saat menjalankan segmentasi.")
            st.code(str(error))
            return

    status_placeholder.empty()

    # Caption sumber gambar
    st.caption(f"Input: `{image_source}`")

    if result.resized:
        st.info(
            "Gambar otomatis diperkecil dari "
            f"{result.original_size[0]}x{result.original_size[1]} menjadi "
            f"{result.processed_size[0]}x{result.processed_size[1]} piksel "
            "agar inferensi Mask R-CNN tetap stabil."
        )

    # Metrics
    unique_labels = len(set(d["label"] for d in result.detections)) if result.detections else 0
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Objek Terdeteksi", len(result.detections))

    with metric_col2:
        st.metric("Kelas Unik", unique_labels)

    with metric_col3:
        st.metric("Waktu Inferensi", f"{result.elapsed_time:.2f} detik")

    # Image comparison
    image_col1, image_col2 = st.columns(2)

    with image_col1:
        st.caption("Original")
        st.image(
            bgr_to_rgb(result.original_image),
            use_container_width=True
        )

    with image_col2:
        st.caption("Instance Segmentation (Mask R-CNN)")
        st.image(
            bgr_to_rgb(result.segmented_image),
            use_container_width=True
        )

    # Processing pipeline - section langsung tanpa expander
    if result.detections:
        bbox_image = draw_bboxes_only(result.original_image, result.detections)

        st.divider()
        st.subheader("Processing Pipeline")

        pipe_col1, pipe_col2, pipe_col3 = st.columns(3)

        with pipe_col1:
            st.caption("Step 1 - Input Image")
            st.image(bgr_to_rgb(result.original_image), use_container_width=True)
            st.markdown(
                "Gambar dibaca sebagai array BGR, lalu dikonversi ke **blob** "
                "berformat `(1, 3, H, W)` menggunakan `cv2.dnn.blobFromImage` "
                "dengan swap BGR ke RGB."
            )

        with pipe_col2:
            st.caption("Step 2 - Bounding Box Detection")
            st.image(bgr_to_rgb(bbox_image), use_container_width=True)
            st.markdown(
                f"Model menghasilkan **100 kandidat** dari forward pass. "
                f"**{len(result.detections)} objek** lolos filter confidence \u2265 0.5. "
                "Bounding box dihitung dari koordinat relatif yang dikembalikan model."
            )

        with pipe_col3:
            st.caption("Step 3 - Instance Mask & Final Output")
            st.image(bgr_to_rgb(result.segmented_image), use_container_width=True)
            st.markdown(
                "Untuk setiap objek: mask kelas di-*resize* ke ukuran bounding box, "
                "di-*threshold* > 0.3 menjadi binary mask, lalu di-*blend* "
                "(`0.4 \u00d7 warna + 0.6 \u00d7 piksel asli`). "
                "Kontur digambar dengan `cv2.findContours` dan label ditambahkan."
            )

    # Detection table
    if result.detections:
        st.divider()
        st.subheader("Detected Objects")
        table_data = [
            {
                "Label": d["label"],
                "Confidence": d["confidence"],
            }
            for d in result.detections
        ]
        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Tidak ada objek yang melewati confidence threshold (0.5).")



    # COCO class list
    with st.expander("Kelas yang Dapat Dideteksi (90 Kelas COCO)", expanded=False):
        st.markdown(
            "Model Mask R-CNN ini dilatih pada dataset **COCO** dan mampu mendeteksi **90 kelas objek** berikut:"
        )
        coco_categories = {
            "Manusia": ["person"],
            "Kendaraan": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
            "Rambu & Fasilitas Jalan": ["traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench"],
            "Hewan": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "Aksesori": ["hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase"],
            "Olahraga": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
            "Peralatan Makan & Dapur": ["bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
            "Makanan": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
            "Furnitur & Ruangan": ["chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door"],
            "Elektronik": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender"],
            "Barang Lainnya": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
        }
        rows = []
        for category, items in coco_categories.items():
            rows.append({"Kategori": category, "Kelas yang Dapat Dideteksi": ", ".join(items)})
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Info dropdown di bawah tabel
    render_info_expander()

    st.markdown(
        """
        <div class="footer-custom">
            Deep Instance Segmentation &nbsp;|&nbsp;
            Pretrained Model: Mask R-CNN Inception V2 (COCO 2018) &nbsp;|&nbsp;
            Lukas Austin &mdash; 140810230011
        </div>
        """,
        unsafe_allow_html=True
    )

main()
