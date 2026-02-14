import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
import threading
import os
import pandas as pd
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import ast

# Configure appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green") 

class NIDSDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Network Intrusion Detection System")
        self.geometry("1400x900")
        self.resizable(True, True)

        # Configure grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.container = ctk.CTkFrame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        
        for F in (MainPage, TrainDataPage, AnalyseFlowPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

class MainPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#2b2b2b")
        self.main_frame.grid(row=0, column=0, padx=40, pady=40, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="Network Intrusion Detection System", 
            font=("Roboto Medium", 36),
            text_color="#FFFFFF"
        )
        self.title_label.pack(pady=(60, 40))

        # Upload Section
        self.upload_card = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#3a3a3a", border_width=1, border_color="#555555")
        self.upload_card.pack(pady=20, padx=100, fill="x")
        
        self.upload_icon_label = ctk.CTkLabel(self.upload_card, text="📂", font=("Segoe UI Emoji", 48))
        self.upload_icon_label.pack(pady=(30, 10))
        
        self.upload_label = ctk.CTkLabel(self.upload_card, text="Upload your CSV file here", font=("Roboto", 18, "bold"))
        self.upload_label.pack(pady=(0, 20))
        
        self.upload_btn = ctk.CTkButton(
            self.upload_card, 
            text="Choose File", 
            command=self.upload_file,
            width=200,
            height=50,
            font=("Roboto", 16, "bold"),
            fg_color="#E04F5F", 
            hover_color="#ff7b89",
            corner_radius=25
        )
        self.upload_btn.pack(pady=(0, 30))
        
        self.file_path_label = ctk.CTkLabel(self.upload_card, text="No file selected", text_color="#aaaaaa")
        self.file_path_label.pack(pady=(0, 20))

        # Action Buttons
        self.btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.btn_frame.pack(pady=50)

        self.train_btn = ctk.CTkButton(
            self.btn_frame, 
            text="Train Data", 
            command=self.start_training,
            width=220,
            height=60,
            font=("Roboto", 18, "bold"),
            fg_color="#F5C542",
            text_color="black",
            hover_color="#ffe082",
            corner_radius=30
        )
        self.train_btn.grid(row=0, column=0, padx=30)

        self.analyse_btn = ctk.CTkButton(
            self.btn_frame, 
            text="Analyse Flow", 
            command=self.start_analysis,
            width=220,
            height=60,
            font=("Roboto", 18, "bold"),
            fg_color="#4CA4F5",
            text_color="white",
            hover_color="#82c4ff",
            corner_radius=30
        )
        self.analyse_btn.grid(row=0, column=1, padx=30)
        
        self.status_label = ctk.CTkLabel(self.main_frame, text="", font=("Roboto", 14))
        self.status_label.pack(pady=20)

        self.selected_file = None

    def upload_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.selected_file = filename
            self.file_path_label.configure(text=f"Selected: {os.path.basename(filename)}")
            try:
                os.makedirs("data", exist_ok=True)
                dest = os.path.join("data", "probe_flow.csv")
                import shutil
                shutil.copy(filename, dest)
                self.status_label.configure(text="File uploaded successfully", text_color="#4CAF50")
            except Exception as e:
                self.status_label.configure(text=f"Error: {e}", text_color="#FF5252")

    def start_training(self):
        self.status_label.configure(text="Training in progress...", text_color="#F5C542")
        self.train_btn.configure(state="disabled")
        threading.Thread(target=self.run_training_script, daemon=True).start()

    def run_training_script(self):
        try:
            # We will read the output from outputs/nids_results.txt if the script writes there
            # But the script also prints to stdout because of how we call it? 
            # Actually train_nids.py redirects stdout to outputs/nids_results.txt internally.
            # So we should just run it and then read the file.
            
            process = subprocess.Popen(
                [sys.executable, "train_nids.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            
            # Read prediction file if exists
            output_file = "outputs/nids_results.txt"
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    full_output = f.read()
            else:
                full_output = stdout + "\n" + stderr

            if process.returncode == 0:
                self.after(0, lambda: self.training_complete(True, full_output))
            else:
                self.after(0, lambda: self.training_complete(False, stderr + "\n" + stdout))
        except Exception as e:
            self.after(0, lambda: self.training_complete(False, str(e)))

    def training_complete(self, success, message):
        self.train_btn.configure(state="normal")
        if success:
            self.status_label.configure(text="Training Complete", text_color="#4CAF50")
            self.controller.frames["TrainDataPage"].update_content(message)
            self.controller.show_frame("TrainDataPage")
        else:
            self.status_label.configure(text="Training Failed", text_color="#FF5252")
            messagebox.showerror("Error", message)

    def start_analysis(self):
        if not self.selected_file and not os.path.exists("data/probe_flow.csv"):
            messagebox.showwarning("Warning", "Upload CSV first!")
            return

        self.status_label.configure(text="Analyzing...", text_color="#4CA4F5")
        self.analyse_btn.configure(state="disabled")
        threading.Thread(target=self.run_analysis_script, daemon=True).start()

    def run_analysis_script(self):
        try:
            script_path = os.path.join("inference", "explain_flow.py")
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.after(0, lambda: self.analysis_complete(True, stdout))
            else:
                self.after(0, lambda: self.analysis_complete(False, stderr))
        except Exception as e:
            self.after(0, lambda: self.analysis_complete(False, str(e)))

    def analysis_complete(self, success, message):
        self.analyse_btn.configure(state="normal")
        if success:
            self.status_label.configure(text="Analysis Complete", text_color="#4CAF50")
            self.controller.frames["AnalyseFlowPage"].update_content(message)
            self.controller.show_frame("AnalyseFlowPage")
        else:
            self.status_label.configure(text="Analysis Failed", text_color="#FF5252")
            messagebox.showerror("Error", message)


class CircularGauge(ctk.CTkCanvas):
    def __init__(self, parent, size=150, **kwargs):
        super().__init__(parent, width=size, height=size, highlightthickness=0, **kwargs)
        self.size = size
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def draw(self, percentage, color, text):
        self.delete("all")

        cx, cy = self.size/2, self.size/2

    # ✅ Background circle (THIS FIXES WHITE ISSUE)
        self.create_oval(
            25, 25,
            self.size - 25, self.size - 25,
            fill="#1E1E1E",
            outline=""
        )

    # Background ring
        self.create_arc(
            15, 15,
            self.size - 15, self.size - 15,
            start=0, extent=359,
            outline="#404040",
            width=15,
            style="arc"
        )

    # Value ring
        extent = -(percentage / 100.0) * 359
        self.create_arc(
            15, 15,
            self.size - 15, self.size - 15,
            start=90, extent=extent,
            outline=color,
            width=15,
            style="arc"
        )

    # Center text
        self.create_text(
            cx, cy,
            text=text,
            fill="white",
            font=("Roboto", 20, "bold")
        )

        
    def on_enter(self, event):
        # Scale up slightly or glow effect? 
        # Canvas modification is tricky, maybe just change outline color slightly?
        pass # Placeholder for advanced effects
        
    def on_leave(self, event):
        pass


class TrainDataPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        self.header = ctk.CTkFrame(self, height=80, fg_color="#2b2b2b", corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew")
        
        self.back_btn = ctk.CTkButton(
            self.header, 
            text="← Back", 
            command=lambda: controller.show_frame("MainPage"), 
            width=100, 
            fg_color="transparent", 
            border_width=1, 
            text_color="white",
            hover_color="#444444"
        )
        self.back_btn.pack(side="left", padx=20, pady=20)
        
        self.title_label = ctk.CTkLabel(self.header, text="Train Data Result", font=("Roboto", 24, "bold"))
        self.title_label.pack(side="left", padx=20)

        # Scrollable Content
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        
    def update_content(self, text_output):
        # Clear previous widgets in scroll_frame
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        # Parse output for each model
        # Pattern: Results for <ModelName>: ... Accuracy: <val> ... Classification Report: ... Confusion Matrix:
        model_blocks = re.split(r"Results for ", text_output)
        
        for block in model_blocks:
            if not block.strip(): continue
            if "Accuracy:" not in block: continue # Skip prologue if any
            
            # Extract header info
            lines = block.split('\n')
            model_name = lines[0].strip().replace(":", "")
            
            accuracy = 0.0
            acc_match = re.search(r"Accuracy:\s*([\d\.]+)", block)
            if acc_match:
                accuracy = float(acc_match.group(1)) * 100
                
            self.create_model_card(model_name, accuracy, block)

    def create_model_card(self, model_name, accuracy, full_text):
        card = ctk.CTkFrame(self.scroll_frame, corner_radius=15, fg_color="#333333", border_width=1, border_color="#555555")
        card.pack(fill="x", pady=15, padx=10, ipady=10)
        
        # Header
        ctk.CTkLabel(card, text=model_name, font=("Roboto", 20, "bold"), text_color="#F5C542").pack(pady=10)
        
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="x", padx=10)
        
        # Left: Gauge
        left_panel = ctk.CTkFrame(content_frame, fg_color="transparent")
        left_panel.pack(side="left", padx=20)
        
        gauge = CircularGauge(left_panel, size=140)
        gauge.pack()
        gauge.draw(accuracy, "#4CA4F5", f"{accuracy:.1f}%")
        ctk.CTkLabel(left_panel, text="Accuracy", font=("Roboto", 14)).pack(pady=5)
        
        # Center: Matrix Plot (Reconstruct from text)
        center_panel = ctk.CTkFrame(content_frame, fg_color="transparent")
        center_panel.pack(side="left", padx=20, expand=True, fill="both")
        
        # Parse Matrix
        try:
            # Find the matrix part
            matrix_str = re.search(r"Confusion Matrix:\s*\[([\s\S]+?)\]\]", full_text)
            if matrix_str:
                mat_text = "[" + matrix_str.group(1) + "]]"
                # Using ast.literal_eval might be risky if format isn't perfect python list
                # Let's try to massage it
                mat_text = mat_text.replace("\n", " ").replace("  ", " ")
                # Remove spaces between brackets? numpy regex parsing is safer?
                # Simple parsing manually
                rows = re.findall(r"\[([\d\s]+)\]", mat_text)
                matrix_data = []
                for r in rows:
                    matrix_data.append([int(x) for x in r.strip().split()])
                
                # Plot
                fig = plt.Figure(figsize=(4, 3.5), dpi=80, facecolor="#333333")
                ax = fig.add_subplot(111)
                ax.set_facecolor("#333333")
                
                # Dark theme plot
                cax = ax.matshow(matrix_data, cmap='Blues')
                # fig.colorbar(cax)
                
                for (i, j), z in np.ndenumerate(matrix_data):
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='black' if z > np.max(matrix_data)/2 else 'white')
                
                ax.set_title("Confusion Matrix", color="white")
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                
                canvas = FigureCanvasTkAgg(fig, master=center_panel)
                canvas.draw()
                canvas.get_tk_widget().pack()
                
        except Exception as e:
            ctk.CTkLabel(center_panel, text=f"[Matrix Parse Error: {e}]", text_color="red").pack()

        # Right: Report
        right_panel = ctk.CTkFrame(content_frame, fg_color="transparent")
        right_panel.pack(side="right", padx=20, fill="y")
        
        # Extract classification report
        report_match = re.search(r"Classification Report:\n([\s\S]+?)(?=Confusion Matrix)", full_text)
        report_text = report_match.group(1) if report_match else "No report found."
        
        textbox = ctk.CTkTextbox(right_panel, width=300, height=200, font=("Consolas", 10))
        textbox.pack(fill="both")
        textbox.insert("0.0", report_text)
        textbox.configure(state="disabled")


class AnalyseFlowPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, height=80, fg_color="#2b2b2b", corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        
        ctk.CTkButton(
            header, text="← Back", command=lambda: controller.show_frame("MainPage"), 
            width=100, fg_color="transparent", border_width=1, hover_color="#444444"
        ).pack(side="left", padx=20, pady=20)
        
        ctk.CTkLabel(header, text="Flow Analysis Result", font=("Roboto", 24, "bold")).pack(side="left", padx=20)

        # Content
        self.content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        
        # Banner
        self.banner = ctk.CTkFrame(self.content, height=60, corner_radius=15, fg_color="#444444")
        self.banner.pack(fill="x", pady=(0, 20))
        self.banner_label = ctk.CTkLabel(self.banner, text="DETECTED TRAFFIC: SCANNING...", font=("Roboto", 22, "bold"))
        self.banner_label.place(relx=0.5, rely=0.5, anchor="center")

        # Gauges Container
        self.gauges_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.gauges_frame.pack(fill="x", pady=20)
        self.gauges_frame.grid_columnconfigure(0, weight=1)
        self.gauges_frame.grid_columnconfigure(1, weight=1)

        # Risk Card
        self.risk_card = self.create_card(self.gauges_frame, "Risk Assessment", 0)
        self.risk_gauge = CircularGauge(self.risk_card, size=160, bg="#2B2B2B")
        self.risk_gauge.pack(pady=15)
        self.risk_text = ctk.CTkLabel(self.risk_card, text="Level: --", font=("Roboto", 16))
        self.risk_text.pack()
        self.priority_text = ctk.CTkLabel(self.risk_card, text="Priority: --", font=("Roboto", 14, "italic"), text_color="#aaaaaa")
        self.priority_text.pack(pady=5)
        
        # Priority/Rec Box
        self.rec_box = ctk.CTkTextbox(self.risk_card, height=80, fg_color="#202020", text_color="#dddddd", font=("Roboto", 12))
        self.rec_box.pack(fill="x", padx=15, pady=10)

        # Confidence Card
        self.conf_card = self.create_card(self.gauges_frame, "Confidence Diagnostics", 1)
        self.conf_gauge = CircularGauge(self.conf_card, size=160, bg="#2B2B2B")
        self.conf_gauge.pack(pady=15)
        self.conf_text = ctk.CTkLabel(self.conf_card, text="Confidence: --%", font=("Roboto", 16))
        self.conf_text.pack()
        self.anomaly_text = ctk.CTkLabel(self.conf_card, text="Anomaly: --", font=("Roboto", 14))
        self.anomaly_text.pack(pady=5)
        
        self.diag_box = ctk.CTkTextbox(self.conf_card, height=80, fg_color="#202020", text_color="#dddddd", font=("Roboto", 12))
        self.diag_box.pack(fill="x", padx=15, pady=10)

        # Action Plan
        self.action_frame = ctk.CTkFrame(self.content, corner_radius=15, fg_color="#2B2B2B")
        self.action_frame.pack(fill="both", pady=20, ipadx=10, ipady=10)
        
        ctk.CTkLabel(self.action_frame, text="Action Plan", font=("Roboto", 20, "bold"), text_color="#4CA4F5").pack(anchor="w", padx=20, pady=10)
        
        # Description
        self.desc_label = ctk.CTkLabel(self.action_frame, text="Description", font=("Roboto", 14, "bold"), text_color="#bbbbbb")
        self.desc_label.pack(anchor="w", padx=20)
        self.desc_text = ctk.CTkLabel(self.action_frame, text="--", font=("Roboto", 13), wraplength=1000, justify="left")
        self.desc_text.pack(anchor="w", padx=20, pady=(0, 15))

        # Do's and Don'ts grid
        self.dd_grid = ctk.CTkFrame(self.action_frame, fg_color="transparent")
        self.dd_grid.pack(fill="x", padx=20)
        self.dd_grid.grid_columnconfigure(0, weight=1)
        self.dd_grid.grid_columnconfigure(1, weight=1)
        
        self.dos_frame = ctk.CTkFrame(self.dd_grid, fg_color="#1e2e1e", corner_radius=10)
        self.dos_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        ctk.CTkLabel(self.dos_frame, text="✅ Do's", font=("Roboto", 16, "bold"), text_color="#4CAF50").pack(pady=10)
        self.dos_text = ctk.CTkTextbox(self.dos_frame, height=100, fg_color="transparent", font=("Roboto", 12))
        self.dos_text.pack(fill="both", padx=10, pady=(0, 10))

        self.donts_frame = ctk.CTkFrame(self.dd_grid, fg_color="#2e1e1e", corner_radius=10)
        self.donts_frame.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
        ctk.CTkLabel(self.donts_frame, text="❌ Don'ts", font=("Roboto", 16, "bold"), text_color="#F44336").pack(pady=10)
        self.donts_text = ctk.CTkTextbox(self.donts_frame, height=100, fg_color="transparent", font=("Roboto", 12))
        self.donts_text.pack(fill="both", padx=10, pady=(0, 10))

    def create_card(self, parent, title, col_idx):
        card = ctk.CTkFrame(parent, corner_radius=20, fg_color="#2B2B2B")
        card.grid(row=0, column=col_idx, padx=15, sticky="nsew")
        ctk.CTkLabel(card, text=title, font=("Roboto", 18, "bold"), text_color="#aaaaaa").pack(pady=15)
        return card

    def update_content(self, text_output):
        # Parsing Logic
        data = {
            "traffic": "Unknown",
            "risk_level": "Unknown",
            "confidence": 0.0,
            "confidence_level": "Low",
            "priority": "Unknown",
            "anomaly": "Unknown",
            "desc": "",
            "dos": "",
            "donts": "",
            "rec": "",
            "diagnostics": "",
            "timestamp": ""
        }
        
        lines = text_output.split('\n')
        diag_lines = []
        capture_dos = False
        capture_donts = False
        
        for line in lines:
            l = line.strip()
            if l.startswith("Detected Traffic Type :"):
                data["traffic"] = l.split(":", 1)[1].strip()
            elif l.startswith("Risk Level"):
                data["risk_level"] = l.split(":", 1)[1].strip()
            elif l.startswith("Model Confidence"):
                # "Model Confidence : 99.9% (High)"
                part = l.split(":", 1)[1].strip()
                try:
                    data["confidence"] = float(part.split("%")[0])
                    if "(" in part:
                         data["confidence_level"] = part.split("(")[1].replace(")", "")
                except: pass
            elif l.startswith("Priority"):
                data["priority"] = l.split(":", 1)[1].strip()
            elif l.startswith("Recommended:"):
                data["rec"] = l.split(":", 1)[1].strip()
            elif l.startswith("Anomaly Status"):
                data["anomaly"] = l.split(":", 1)[1].strip()
            elif l.startswith("Description:"):
                data["desc"] = l.split(":", 1)[1].strip()
            elif l.startswith("Do:"):
                data["dos"] = l.split(":", 1)[1].strip()
            elif l.startswith("Don't:"):
                data["donts"] = l.split(":", 1)[1].strip()
            elif l.startswith("Timestamp"):
                data["timestamp"] = l.split(":", 1)[1].strip()
            elif "Class Probabilities" in l:
                diag_lines.append(l)
            elif "Raw Confidence" in l or "Calibrated Confidence" in l:
                diag_lines.append(l)

        # Update UI
        
        # Banner Color
        traffic = data["traffic"].upper()
        if "BENIGN" in traffic or "NORMAL" in traffic:
             self.banner.configure(fg_color="#2E7D32") # Green
        else:
             self.banner.configure(fg_color="#C62828") # Red
        self.banner_label.configure(text=f"DETECTED TRAFFIC: {traffic}")

        # Risk Gauge
        risk = data["risk_level"]
        r_color = "#00FF00" # Low
        r_val = 25
        if "Medium" in risk: 
            r_color = "#FFFF00"
            r_val = 50
        elif "High" in risk: 
            r_color = "#FFA500"
            r_val = 75
        elif "Critical" in risk: 
            r_color = "#FF0000"
            r_val = 100
            
        self.risk_gauge.configure(bg=self.risk_card._apply_appearance_mode(self.risk_card._fg_color))
        self.risk_gauge.draw(r_val, r_color, risk)
        self.risk_text.configure(text=f"Level: {risk}", text_color=r_color)
        self.priority_text.configure(text=f"Priority: {data['priority']}")
        self.rec_box.delete("0.0", "end")
        self.rec_box.insert("0.0", f"Rec: {data['rec']}")

        # Confidence Gauge
        conf = data["confidence"]
        c_color = "#FF0000"
        if conf >= 85: c_color = "#00FF00"
        elif conf >= 70: c_color = "#FFFF00"
        
        self.conf_gauge.configure(bg=self.conf_card._apply_appearance_mode(self.conf_card._fg_color))
        self.conf_gauge.draw(conf, c_color, f"{conf:.1f}%")
        self.conf_text.configure(text=f"Confidence: {conf:.1f}%", text_color=c_color)
        self.anomaly_text.configure(text=f"Anomaly: {data['anomaly']}")
        
        diag_str = "\n".join(diag_lines)
        self.diag_box.delete("0.0", "end")
        self.diag_box.insert("0.0", diag_str)

        # Action Plan
        self.desc_text.configure(text=data["desc"])
        self.dos_text.delete("0.0", "end")
        self.dos_text.insert("0.0", data["dos"])
        self.donts_text.delete("0.0", "end")
        self.donts_text.insert("0.0", data["donts"])


if __name__ == "__main__":
    app = NIDSDashboard()
    app.mainloop()
