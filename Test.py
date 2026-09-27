import tkinter as tk
from tkinter import filedialog, ttk
import os
#os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import time
import threading
import joblib
from genre_features import extract_features


class MusicPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Player")
        self.root.geometry("600x500")
        self.root.configure(bg="#1a1a1a")

        pygame.mixer.init()

        self.playlist = []
        self.current_track = 0
        self.is_playing = False
        self.is_paused = False
        self.song_length = 0

        self.genre_model = None
        self.genre_scaler = None
        self.genre_cache = {}  # file path -> (genre, confidence)
        self.load_genre_model()

        self.create_widgets()

    def load_genre_model(self, path="genre_model.pkl"):
        try:
            data = joblib.load(path)
            self.genre_model = data["model"]
            self.genre_scaler = data["scaler"]
        except FileNotFoundError:
            print(f"No genre model found at {path}. "
                  "Run train_genre_classifier.py to create one.")
            self.genre_model = None
            self.genre_scaler = None

    def create_widgets(self):
        # Create main canvas and scrollbars
        main_canvas = tk.Canvas(self.root, bg="#1a1a1a", highlightthickness=0)
        main_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        main_hscrollbar = ttk.Scrollbar(self.root, orient="horizontal", command=main_canvas.xview)

        # Create a frame inside the canvas
        self.scrollable_frame = tk.Frame(main_canvas, bg="#1a1a1a")

        # Configure the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        # Create window with center anchor and bind to canvas width
        canvas_window = main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set, xscrollcommand=main_hscrollbar.set)

        # Let the scrollable frame fill the canvas width, but grow wider if its
        # natural (requested) width exceeds the canvas — that's what makes
        # horizontal scrolling actually kick in when content needs more space.
        def configure_scroll_frame(event):
            frame_width = max(event.width, self.scrollable_frame.winfo_reqwidth())
            main_canvas.itemconfig(canvas_window, width=frame_width)

        main_canvas.bind("<Configure>", configure_scroll_frame)

        # Pack the scrollbars and canvas (bottom/right scrollbars packed first
        # so they aren't squeezed out by the expanding canvas)
        main_hscrollbar.pack(side="bottom", fill="x")
        main_scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel to canvas (vertical scroll; hold Shift for horizontal)
        main_canvas.bind_all("<MouseWheel>", lambda e: main_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        main_canvas.bind_all("<Shift-MouseWheel>",
                             lambda e: main_canvas.xview_scroll(int(-1 * (e.delta / 120)), "units"))

        # Title
        title = tk.Label(self.scrollable_frame, text="🎵 Music Player", font=("Arial", 20, "bold"),
                         bg="#1a1a1a", fg="#ffffff")
        title.pack(pady=20)

        # Now Playing
        self.now_playing = tk.Label(self.scrollable_frame, text="No track loaded",
                                    font=("Arial", 12), bg="#1a1a1a", fg="#00ff88")
        self.now_playing.pack(pady=10)

        # Genre
        self.genre_label = tk.Label(self.scrollable_frame, text="Genre: —",
                                    font=("Arial", 11), bg="#1a1a1a", fg="#00aaff")
        self.genre_label.pack(pady=(0, 10))

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(self.scrollable_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      variable=self.progress_var, command=self.seek_music)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)

        # Time Labels
        time_frame = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
        time_frame.pack(fill=tk.X, padx=20)

        self.current_time = tk.Label(time_frame, text="0:00", bg="#1a1a1a", fg="#888888")
        self.current_time.pack(side=tk.LEFT)

        self.total_time = tk.Label(time_frame, text="0:00", bg="#1a1a1a", fg="#888888")
        self.total_time.pack(side=tk.RIGHT)

        # Control Buttons
        control_frame = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
        control_frame.pack(pady=20)

        btn_style = {"font": ("Arial", 12), "bg": "#333333", "fg": "#ffffff",
                     "width": 12, "relief": tk.FLAT, "cursor": "hand2"}

        self.prev_btn = tk.Button(control_frame, text="⏮ Previous", command=self.previous_track, **btn_style)
        self.prev_btn.grid(row=0, column=0, padx=5)

        self.play_btn = tk.Button(control_frame, text="▶ Play", command=self.play_music, **btn_style)
        self.play_btn.grid(row=0, column=1, padx=5)

        self.pause_btn = tk.Button(control_frame, text="⏸ Pause", command=self.pause_music, **btn_style)
        self.pause_btn.grid(row=0, column=2, padx=5)

        self.next_btn = tk.Button(control_frame, text="Next ⏭", command=self.next_track, **btn_style)
        self.next_btn.grid(row=0, column=3, padx=5)

        # Playlist
        playlist_label = tk.Label(self.scrollable_frame, text="Playlist", font=("Arial", 14, "bold"),
                                  bg="#1a1a1a", fg="#ffffff")
        playlist_label.pack(pady=(20, 5))

        # Listbox with Scrollbar
        list_frame = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.playlist_box = tk.Listbox(list_frame, bg="#2a2a2a", fg="#ffffff",
                                       selectbackground="#00ff88", selectforeground="#000000",
                                       font=("Arial", 10), yscrollcommand=scrollbar.set,
                                       relief=tk.FLAT, height=8)
        self.playlist_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.playlist_box.bind('<Double-Button-1>', self.play_selected)

        scrollbar.config(command=self.playlist_box.yview)

        # Add/Remove Buttons
        file_frame = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
        file_frame.pack(pady=10)

        add_btn = tk.Button(file_frame, text="➕ Add Songs", command=self.add_songs, **btn_style)
        add_btn.grid(row=0, column=0, padx=5)

        remove_btn = tk.Button(file_frame, text="➖ Remove", command=self.remove_song, **btn_style)
        remove_btn.grid(row=0, column=1, padx=5)

        classify_btn = tk.Button(file_frame, text="🎧 Classify Genre",
                                 command=self.classify_current_genre, **btn_style)
        classify_btn.grid(row=0, column=2, padx=5)

        # Volume Control
        vol_frame = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
        vol_frame.pack(pady=10)

        vol_label = tk.Label(vol_frame, text="🔊 Volume:", bg="#1a1a1a", fg="#ffffff")
        vol_label.pack(side=tk.LEFT, padx=5)

        self.volume_var = tk.DoubleVar(value=70)
        volume_slider = ttk.Scale(vol_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                  variable=self.volume_var, command=self.change_volume)
        volume_slider.pack(side=tk.LEFT, padx=5)

        self.update_progress()

    def add_songs(self):
        files = filedialog.askopenfilenames(title="Select Music Files",
                                            filetypes=(("MP3 Files", "*.mp3"), ("All Files", "*.*")))
        for file in files:
            self.playlist.append(file)
            self.playlist_box.insert(tk.END, os.path.basename(file))

    def remove_song(self):
        try:
            selected = self.playlist_box.curselection()[0]
            self.playlist_box.delete(selected)
            del self.playlist[selected]
        except IndexError:
            pass

    def play_selected(self, event):
        try:
            self.current_track = self.playlist_box.curselection()[0]
            self.play_music()
        except IndexError:
            pass

    def play_music(self):
        if not self.playlist:
            return

        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.is_playing = True
        else:
            try:
                pygame.mixer.music.load(self.playlist[self.current_track])
                pygame.mixer.music.play()
                self.is_playing = True
                self.is_paused = False

                # Get song length
                audio = MP3(self.playlist[self.current_track])
                self.song_length = audio.info.length

                # Update display
                song_name = os.path.basename(self.playlist[self.current_track])
                self.now_playing.config(text=f"♪ {song_name}")
                self.playlist_box.selection_clear(0, tk.END)
                self.playlist_box.selection_set(self.current_track)

                self.show_cached_or_classify(self.playlist[self.current_track])

            except Exception as e:
                print(f"Error playing track: {e}")

    def show_cached_or_classify(self, track_path):
        """Show a genre immediately if we've already classified this file,
        otherwise kick off classification in the background."""
        if track_path in self.genre_cache:
            genre, confidence = self.genre_cache[track_path]
            self.genre_label.config(text=f"Genre: {genre.capitalize()} ({confidence:.0f}%)")
        else:
            self.genre_label.config(text="Genre: —")
            self.classify_current_genre()

    def classify_current_genre(self):
        if not self.playlist:
            return
        if self.genre_model is None:
            self.genre_label.config(text="Genre: model not loaded")
            return

        track_path = self.playlist[self.current_track]
        self.genre_label.config(text="Genre: analyzing…")
        threading.Thread(target=self._classify_worker, args=(track_path,), daemon=True).start()

    def _classify_worker(self, track_path):
        try:
            feats = extract_features(track_path).reshape(1, -1)
            feats_scaled = self.genre_scaler.transform(feats)
            genre = self.genre_model.predict(feats_scaled)[0]
            probs = self.genre_model.predict_proba(feats_scaled)[0]
            confidence = max(probs) * 100

            self.genre_cache[track_path] = (genre, confidence)

            def update_label():
                # Only update if this track is still the one playing
                if self.playlist[self.current_track] == track_path:
                    self.genre_label.config(text=f"Genre: {genre.capitalize()} ({confidence:.0f}%)")

            self.root.after(0, update_label)
        except Exception as e:
            self.root.after(0, lambda: self.genre_label.config(text=f"Genre: error ({e})"))

    def pause_music(self):
        if self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True

    def next_track(self):
        if self.playlist:
            self.current_track = (self.current_track + 1) % len(self.playlist)
            self.play_music()

    def previous_track(self):
        if self.playlist:
            self.current_track = (self.current_track - 1) % len(self.playlist)
            self.play_music()

    def change_volume(self, val):
        volume = float(val) / 100
        pygame.mixer.music.set_volume(volume)

    def seek_music(self, val):
        if self.is_playing and self.song_length > 0:
            seek_pos = (float(val) / 100) * self.song_length
            pygame.mixer.music.play(start=seek_pos)

    def update_progress(self):
        if self.is_playing and not self.is_paused:
            try:
                current_pos = pygame.mixer.music.get_pos() / 1000
                if self.song_length > 0:
                    progress = (current_pos / self.song_length) * 100
                    self.progress_var.set(progress)

                    # Update time labels
                    self.current_time.config(text=self.format_time(current_pos))
                    self.total_time.config(text=self.format_time(self.song_length))

                # Check if song ended
                if not pygame.mixer.music.get_busy():
                    self.next_track()
            except:
                pass

        self.root.after(100, self.update_progress)

    def format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"


if __name__ == "__main__":
    root = tk.Tk()
    player = MusicPlayer(root)
    root.mainloop()