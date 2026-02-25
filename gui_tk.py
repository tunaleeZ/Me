#!/usr/bin/env python3
from __future__ import annotations

import json
import queue
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Optional

import paho.mqtt.client as mqtt

from common.utils import now_iso
from modules.gesture.config_gesture import settings


@dataclass
class WaypointState:
    order: tuple[str, ...] = ("A", "B", "C")
    index: int = 0

    def expected(self) -> Optional[str]:
        if self.index >= len(self.order):
            return None
        return self.order[self.index]

    def consume(self, label: str) -> tuple[bool, str]:
        exp = self.expected()
        if exp is None:
            return False, "Sequence completed. Press Reset to start again."
        if label != exp:
            return False, f"Invalid order. Expected '{exp}' next."

        self.index += 1
        if self.index >= len(self.order):
            return True, "A -> B -> C completed."
        return True, f"Point {label} locked. Next: {self.expected()}"

    def segment_for(self, label: str) -> str:
        if self.index == 1:
            return ""
        prev = self.order[self.index - 2]
        return f"{prev}{label}"

    def reset(self) -> None:
        self.index = 0


class GestureTkApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Gesture Waypoint Controller")
        self.root.configure(bg="#0b1118")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _e: self.close())

        self.msg_queue: "queue.Queue[str]" = queue.Queue()
        self.state = WaypointState()
        self.camera_sizes = ["640x480", "800x600", "1280x720"]
        self.camera_size_var = tk.StringVar(value="640x480")

        self.client = mqtt.Client(client_id=f"{settings.mqtt_client_id}.tk")
        if settings.mqtt_user:
            self.client.username_pw_set(settings.mqtt_user, settings.mqtt_pass)
        if settings.mqtt_ssl:
            self.client.tls_set()

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

        self._build_ui()
        self._connect_mqtt()

        self.root.after(100, self._drain_messages)
        self.root.after(250, self._animate_preview)

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        top = tk.Frame(self.root, bg="#0b1118")
        top.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        top.grid_columnconfigure(0, weight=0)
        top.grid_columnconfigure(1, weight=1)
        top.grid_rowconfigure(0, weight=1)

        control = tk.Frame(top, bg="#111827", highlightthickness=1, highlightbackground="#1f2937")
        control.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        tk.Label(
            control,
            text="Waypoint",
            fg="#f3f4f6",
            bg="#111827",
            font=("Segoe UI", 24, "bold"),
        ).pack(padx=16, pady=(16, 8), anchor="w")

        self.status_var = tk.StringVar(value="Ready. Press A to start.")
        tk.Label(
            control,
            textvariable=self.status_var,
            fg="#93c5fd",
            bg="#111827",
            justify="left",
            wraplength=260,
            font=("Segoe UI", 11),
        ).pack(padx=16, pady=(0, 10), anchor="w")

        self._style_buttons()
        self.btn_a = ttk.Button(control, text="A", command=lambda: self._on_lock("A"))
        self.btn_b = ttk.Button(control, text="B", command=lambda: self._on_lock("B"))
        self.btn_c = ttk.Button(control, text="C", command=lambda: self._on_lock("C"))

        self.btn_a.pack(fill="x", padx=16, ipady=12)
        tk.Label(control, text="v", fg="#22d3ee", bg="#111827", font=("Segoe UI", 18, "bold")).pack(pady=2)
        self.btn_b.pack(fill="x", padx=16, ipady=12)
        tk.Label(control, text="v", fg="#22d3ee", bg="#111827", font=("Segoe UI", 18, "bold")).pack(pady=2)
        self.btn_c.pack(fill="x", padx=16, ipady=12)

        actions = tk.Frame(control, bg="#111827")
        actions.pack(fill="x", padx=16, pady=(14, 16))
        ttk.Button(actions, text="Reset", command=self._reset_sequence).pack(fill="x", ipady=8)
        ttk.Button(actions, text="Exit (Esc)", command=self.close).pack(fill="x", ipady=8, pady=(8, 0))

        camera = tk.Frame(top, bg="#06090f", highlightthickness=1, highlightbackground="#1f2937")
        camera.grid(row=0, column=1, sticky="nsew")
        camera.grid_rowconfigure(1, weight=1)
        camera.grid_columnconfigure(0, weight=1)

        cam_header = tk.Frame(camera, bg="#06090f")
        cam_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        cam_header.grid_columnconfigure(0, weight=1)

        tk.Label(
            cam_header,
            text="Camera View (Demo Layout)",
            fg="#e5e7eb",
            bg="#06090f",
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        size_box = ttk.Combobox(
            cam_header,
            textvariable=self.camera_size_var,
            values=self.camera_sizes,
            state="readonly",
            width=10,
        )
        size_box.grid(row=0, column=1, sticky="e")

        self.preview = tk.Canvas(camera, bg="#000000", highlightthickness=0)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        bottom = tk.Frame(self.root, bg="#0b1118")
        bottom.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        bottom.grid_rowconfigure(1, weight=1)
        bottom.grid_columnconfigure(0, weight=1)

        tk.Label(
            bottom,
            text=f"MQTT PUB/SUB Monitor  |  {settings.mqtt_host}:{settings.mqtt_port}",
            fg="#e5e7eb",
            bg="#0b1118",
            font=("Consolas", 13, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        log_wrap = tk.Frame(bottom, bg="#000000", highlightthickness=1, highlightbackground="#1f2937")
        log_wrap.grid(row=1, column=0, sticky="nsew")
        log_wrap.grid_rowconfigure(0, weight=1)
        log_wrap.grid_columnconfigure(0, weight=1)

        self.log = tk.Text(
            log_wrap,
            bg="#000000",
            fg="#f9fafb",
            insertbackground="#f9fafb",
            font=("Consolas", 11),
            state="disabled",
            wrap="none",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=yscroll.set)

        self._set_button_states()

    def _style_buttons(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Segoe UI", 18, "bold"), padding=6)

    def _animate_preview(self) -> None:
        self.preview.delete("all")
        w = max(10, self.preview.winfo_width())
        h = max(10, self.preview.winfo_height())
        t = int(time.time() * 1000)

        try:
            base_w, base_h = [int(v) for v in self.camera_size_var.get().split("x", 1)]
        except Exception:
            base_w, base_h = 640, 480

        # Fit selected camera size into preview area while preserving aspect ratio.
        fit = min(w / float(base_w), h / float(base_h))
        frame_w = int(base_w * fit)
        frame_h = int(base_h * fit)
        x0 = (w - frame_w) // 2
        y0 = (h - frame_h) // 2
        x1 = x0 + frame_w
        y1 = y0 + frame_h

        x = x0 + 30 + (t // 8) % max(40, frame_w - 60)

        self.preview.create_rectangle(0, 0, w, h, fill="#05070b", outline="")
        self.preview.create_rectangle(x0, y0, x1, y1, fill="#000000", outline="#1f2937")
        self.preview.create_text(
            x0 + 12,
            y0 + 20,
            text=f"OpenCV area: {base_w}x{base_h}",
            fill="#9ca3af",
            anchor="w",
            font=("Segoe UI", 11),
        )
        cy = y0 + frame_h // 2
        self.preview.create_oval(x - 14, cy - 14, x + 14, cy + 14, fill="#22d3ee", outline="")
        self.preview.create_text(
            x1 - 12,
            y1 - 12,
            text=time.strftime("%H:%M:%S"),
            fill="#e5e7eb",
            anchor="se",
            font=("Consolas", 11),
        )

        self.root.after(80, self._animate_preview)

    def _connect_mqtt(self) -> None:
        try:
            self.client.connect(settings.mqtt_host, settings.mqtt_port, 60)
            self.client.loop_start()
            self._append_log(f"[INFO] Connecting to MQTT broker {settings.mqtt_host}:{settings.mqtt_port}")
        except Exception as exc:
            self._append_log(f"[ERROR] MQTT connect failed: {exc}")
            self.status_var.set(f"MQTT connect failed: {exc}")

    def on_connect(self, client, _userdata, _flags, rc, properties=None) -> None:
        if rc == 0:
            client.subscribe(settings.mqtt_reply)
            client.subscribe(settings.mqtt_status)
            self.msg_queue.put(f"[INFO] Connected. Subscribed: {settings.mqtt_reply}, {settings.mqtt_status}")
            self.status_var.set("Connected. Press A -> B -> C.")
        else:
            self.msg_queue.put(f"[ERROR] MQTT connect rc={rc}")
            self.status_var.set(f"MQTT connect failed rc={rc}")

    def on_disconnect(self, _client, _userdata, rc, properties=None) -> None:
        self.msg_queue.put(f"[WARN] Disconnected from MQTT (rc={rc})")

    def on_message(self, _client, _userdata, msg) -> None:
        payload = msg.payload.decode("utf-8", errors="replace")
        self.msg_queue.put(f"[SUB] [{time.strftime('%H:%M:%S')}] {msg.topic}: {payload}")

    def _drain_messages(self) -> None:
        while not self.msg_queue.empty():
            self._append_log(self.msg_queue.get_nowait())
        self.root.after(100, self._drain_messages)

    def _append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_button_states(self) -> None:
        expected = self.state.expected()
        self.btn_a.configure(state=("normal" if expected == "A" else "disabled"))
        self.btn_b.configure(state=("normal" if expected == "B" else "disabled"))
        self.btn_c.configure(state=("normal" if expected == "C" else "disabled"))

    def _build_waypoint_message(self, label: str, segment: str) -> dict:
        return {
            "ok": True,
            "ts": now_iso(),
            "timestamp": time.time(),
            "workflow_type": "pick_place",
            "waypoints": [
                {
                    "label": label,
                    "segment": segment,
                    "position": [],
                    "orientation": [],
                    "gripper_state": "",
                    "velocity": 0.2,
                }
            ],
        }

    def _on_lock(self, label: str) -> None:
        ok, msg = self.state.consume(label)
        if not ok:
            self.status_var.set(msg)
            self._append_log(f"[WARN] {msg}")
            self._set_button_states()
            return

        segment = self.state.segment_for(label)
        body = self._build_waypoint_message(label, segment)
        payload = json.dumps(body, ensure_ascii=False)
        res = self.client.publish(settings.mqtt_reply, payload)

        if getattr(res, "rc", mqtt.MQTT_ERR_SUCCESS) != mqtt.MQTT_ERR_SUCCESS:
            self.status_var.set("Publish failed. Check MQTT connection.")
            self._append_log(f"[ERROR] Publish failed to {settings.mqtt_reply}")
        else:
            self.status_var.set(msg)
            self._append_log(f"[PUB] {settings.mqtt_reply}: {payload}")

        self._set_button_states()

    def _reset_sequence(self) -> None:
        self.state.reset()
        self.status_var.set("Sequence reset. Press A to start.")
        self._append_log("[INFO] Sequence reset")
        self._set_button_states()

    def close(self) -> None:
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = GestureTkApp()
    app.run()


if __name__ == "__main__":
    main()
