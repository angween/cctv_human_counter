import cv2
import numpy as np
from ultralytics import YOLO

def main(save_video=False, show_video=True):
    # 1. Load model YOLOv11 nano
    print("Loading YOLOv11 model...")
    model = YOLO("yolo11n.pt")
    print("Model loaded.")

    # 2. Buka file video
    video_path = "test1.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka {video_path}")
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

    # Definikan koordinat garis batas (misal garis maya di 1/3 bawah layar)
    line_y = height * 2 // 3
    
    # State tracking
    track_history = {} # ID: { "center_y": list_of_y, "counted": boolean, "dir": string }
    
    in_count = 0
    out_count = 0

    print(f"Video resolution: {width}x{height} @ {fps}fps")
    print(f"Counting line Y-axis: {line_y}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video selesai atau error membaca frame.")
            break

        # Gambar garis penghitung
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

        # 3. Jalankan YOLOv11 tracking
        # classes=[0] untuk hanya mendeteksi "person"
        # persist=True agar ID tracking dipertahankan antar frame
        results = model.track(frame, classes=[0], persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Validasi ada deteksi dengan ID
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Pusat bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Jika ID baru muncul
                if track_id not in track_history:
                    track_history[track_id] = {
                        "history": [],
                        "counted": False
                    }
                
                # Simpan histori Y
                hist = track_history[track_id]["history"]
                hist.append(cy)
                
                # Batasi memori, simpan max 30 posisi terakhir
                if len(hist) > 30:
                    hist.pop(0)

                # Cek jika belum dihitung & histori memadai
                if not track_history[track_id]["counted"] and len(hist) > 2:
                    # Ambil posisi Y beberapa frame yg lalu dan Y sekarang
                    prev_y = hist[-2]
                    curr_y = hist[-1]
                    
                    # Logic menyeberang Garis
                    # Dari atas ke bawah (In/Out tergantung perspektif, kita asumsikan atas -> bawah = IN, bawah -> atas = OUT)
                    if prev_y < line_y and curr_y >= line_y:
                        in_count += 1
                        track_history[track_id]["counted"] = True
                    elif prev_y > line_y and curr_y <= line_y:
                        out_count += 1
                        track_history[track_id]["counted"] = True

                # Visualisasi Bounding Box & Titik Pusat
                color = (0, 255, 0) if track_history[track_id]["counted"] else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

            cv2.imshow("CCTV Human Counter", frame_resized)
            
            # Tekan 'Esc' (27) untuk keluar
            if cv2.waitKey(1) & 0xFF == 27:
                print("Esc ditekan. Keluar dari program.")
                break

        # Print progress tiap 30 frame
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
            print(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}... IN: {in_count}, OUT: {out_count}")

    cap.release()
    if save_video and out is not None:
        out.release()
        print(f"Selesai! Video disimpan di {out_video_path}")
    else:
        print("Selesai diproses.")
    
    if show_video:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ganti jadi save_video=True jika ingin menyimpan hasil
    # Ganti jadi show_video=False jika proses ingin berjalan di background tanpa UI
    main(save_video=False, show_video=True)
