import cv2
import numpy as np
import msvcrt
from ultralytics import YOLO

# Workaround: ultralytics monkey-patches cv2.imshow dengan wrapper yang
# bisa error di lingkungan tertentu. Reload cv2 untuk restore fungsi asli.
import importlib
importlib.reload(cv2)

def main(video_source="test1.mp4", save_video=False, show_video=True):
    # 1. Load model YOLOv11 nano
    print("Loading YOLOv11 model...")
    model = YOLO("yolo11n.pt")
    print("Model loaded.")

    # 2. Buka source video (file, RTSP, atau index kamera)
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka {video_source}")
        return

    # Ambil properti video untuk line definition
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup VideoWriter
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = "output_test1.mp4"
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # ROI (Region of Interest) - area pintu masuk
    # Koordinat menggunakan rasio agar dinamis terhadap resolusi
    roi_x1 = int(width * 0.25)   # Batas kiri pintu
    roi_y1 = 50                  # 50px dari atas frame
    roi_x2 = int(width * 0.75)   # Batas kanan pintu
    roi_y2 = int(height * 0.85)   # Batas bawah pintu

    # Garis penghitung horizontal (di dalam ROI)
    # Garis harus cukup rendah agar orang yang keluar (OUT) terdeteksi di Zone 2 (bawah) dulu
    line_y_1 = int(height * 0.4)
    line_y_2 = line_y_1 + 90

    # State tracking
    track_history = {} # ID: { "history": list_of_y, "counted": boolean }
    
    # Short ID mapping (agar ID tampil pendek, di-recycle saat orang hilang)
    id_map = {}           # tracker_id -> short_id
    next_short_id = 1     # ID pendek berikutnya
    recycled_ids = []     # ID pendek yang bisa digunakan ulang
    
    in_count = 0
    out_count = 0
    prev_in_count = 0
    prev_out_count = 0

    print(f"Video resolution: {width}x{height} @ {fps}fps")
    print(f"ROI: ({roi_x1},{roi_y1}) to ({roi_x2},{roi_y2})")
    print(f"Counting lines Y-axis: {line_y_1} and {line_y_2}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video selesai atau error membaca frame.")
            break

        # Gambar ROI (kotak area pintu masuk)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)
        cv2.putText(frame, "DOOR ZONE", (roi_x1, roi_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Gambar garis penghitung horizontal (hanya di dalam ROI)
        cv2.line(frame, (roi_x1, line_y_1), (roi_x2, line_y_1), (0, 255, 255), 2)
        cv2.line(frame, (roi_x1, line_y_2), (roi_x2, line_y_2), (0, 255, 255), 2)

        # 3. Jalankan YOLOv11 tracking
        # classes=[0] untuk hanya mendeteksi "person"
        # persist=True agar ID tracking dipertahankan antar frame
        results = model.track(frame, classes=[0], persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Validasi ada deteksi dengan ID
        active_ids_this_frame = set()
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                
                # Filter: confidence minimal 0.3 (dilonggarkan agar orang yang tertutup motor tetap terdeteksi)
                if conf < 0.3:
                    continue
                
                # Pusat bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Filter: hanya proses deteksi di dalam ROI
                if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                    continue

                # Jika ID baru muncul
                if track_id not in track_history:
                    track_history[track_id] = {
                        "counted": False,
                        "zones": [],
                        "invisible_count": 0
                    }
                    # Beri short ID
                    if recycled_ids:
                        id_map[track_id] = recycled_ids.pop(0)
                    else:
                        id_map[track_id] = next_short_id
                        next_short_id += 1
                        # Reset short ID ke 1 jika sudah mencapai 100
                        if next_short_id >= 100:
                            next_short_id = 1
                            recycled_ids.clear()
                
                active_ids_this_frame.add(track_id)
                track_history[track_id]["invisible_count"] = 0  # Reset counter jika terdeteksi kembali
                short_id = id_map.get(track_id, track_id)
                
                # Penentuan posisi berada di Zone mana (0: Atas, 1: Tengah, 2: Bawah)
                if cy < line_y_1:
                    curr_zone = 0
                elif cy > line_y_2:
                    curr_zone = 2
                else:
                    curr_zone = 1
                    
                # Masukkan ke list zones jika zone berubah
                zones = track_history[track_id]["zones"]
                if not zones or zones[-1] != curr_zone:
                    zones.append(curr_zone)

                # Cek jika belum dihitung & histori lengkap melewati kedua zone
                if not track_history[track_id]["counted"]:
                    if 0 in zones and 2 in zones:
                        # Cari kapan pertama kali dia menginjak Zone 0 dan Zone 2
                        idx_0 = zones.index(0)
                        idx_2 = zones.index(2)
                        
                        # IN: Jika dia menginjak Zone 0 (Atas) lebih dulu daripada Zone 2 (Bawah)
                        if idx_0 < idx_2:
                            in_count += 1
                            print(f"  >> [IN] ID {short_id} zones: {zones}")
                        # OUT: Jika dia menginjak Zone 2 (Bawah) lebih dulu daripada Zone 0 (Atas)
                        else:
                            out_count += 1
                            print(f"  >> [OUT] ID {short_id} zones: {zones}")
                            
                        track_history[track_id]["counted"] = True

                # Visualisasi Bounding Box & Titik Pusat
                color = (0, 255, 0) if track_history[track_id]["counted"] else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, f"ID: {short_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Bersihkan ID yang sudah hilang dari frame (recycle short ID)
        stale_ids = []
        for tid in list(track_history.keys()):
            if tid not in active_ids_this_frame:
                track_history[tid]["invisible_count"] += 1
                if track_history[tid]["invisible_count"] > 30:  # Toleransi hilang hingga 30 frame akibat overlap/oklusi
                    stale_ids.append(tid)

        for tid in stale_ids:
            if tid in id_map:
                recycled_ids.append(id_map.pop(tid))
            del track_history[tid]
        recycled_ids.sort()

        # 4. Tampilkan HUD Teks Masuk/Keluar
        cv2.putText(frame, f"IN: {in_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"OUT: {out_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 5. Simpan frame ke output video
        if save_video and out is not None:
            out.write(frame)

        # 6. Tampilkan frame
        if show_video:
            # Resize frame agar muat di layar (misalnya tinggi 720px)
            display_height = 720
            # Hitung lebar proporsional
            display_width = int(width * (display_height / height))
            frame_resized = cv2.resize(frame, (display_width, display_height))

            cv2.imshow("Customer Counter", frame_resized)
            
            # Tekan 'Esc' (27) untuk keluar
            if cv2.waitKey(1) & 0xFF == 27:
                # print("Esc ditekan. Keluar dari program.")
                break
        else:
            # Cek keyboard input dari terminal (tanpa window)
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x1b':  # Esc key
                    # print("Esc ditekan. Keluar dari program.")
                    break

        # Print hanya saat ada perubahan IN atau OUT
        if in_count != prev_in_count or out_count != prev_out_count:
            if in_count != prev_in_count:
                print(f"[IN] Total IN: {in_count}, OUT: {out_count}")
            if out_count != prev_out_count:
                print(f"[OUT] Total IN: {in_count}, OUT: {out_count}")
            prev_in_count = in_count
            prev_out_count = out_count

    cap.release()
    if save_video and out is not None:
        out.release()
        print(f"Selesai! Video disimpan di {out_video_path}")
    else:
        print("Selesai diproses.")
    
    if show_video:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Konfigurasi source:
    user = "pooling"
    password = "YamahaNo1"
    rtsp_url = f"rtsp://{user}:{password}@172.16.0.187:554/Streaming/Channels/401"  # sentral
    # rtsp_url = f"rtsp://{user}:{password}@172.16.0.196:554/Streaming/Channels/101" # lubeg

    sources = [
        rtsp_url,    # Index 0: CCTV (RTSP)
        0,           # Index 1: Webcam (Internal/USB)
        "test1.mp4"  # Index 2: File Video Lokal
    ]
    
    # Pilih source dengan merubah index di bawah (0, 1, atau 2)
    source_index = 0
    video_input = sources[source_index]
    
    print(f"Starting with source: {video_input}")
    main(video_source=video_input, save_video=False, show_video=True)
