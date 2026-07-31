import cv2
from moviepy import ImageSequenceClip
import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from src.sim import MjcSim

os.environ['QT_QPA_PLATFORM'] = "offscreen"
print(f"Using Qt platform: {os.environ.get('QT_QPA_PLATFORM')}")


class Recorder:
    """Class to record and generate videos from simulation data."""
    def __init__(self, 
                 video_fps: float, 
                 plot_attributes: dict[str, dict[str, str]], 
                 plot_structure: list[list[str]],
                 plot_time_range: list[float, float]=None) -> None:
        """
        - plot_attributes: Dict mapping variable names to {title}.
        - plot_structure: List of subplots, each containing [x, y1, y2, ...].
        """
        # Determine the output interval based on the video fps
        self.video_fps = video_fps
        self.output_interval = 1 / video_fps
        self.next_vid_output_time = 0
        self.plt_data = {}
        self.robot_frames = []
        self.plt_attr = plot_attributes  # { "varName": {"title": "plotTitle", "color": (G,B,R), "line_style": "--"} }
        self.plt_struc = plot_structure  # [[xvar, yvar1, yvar2], [xvar2, yvar3], ...]
        self.plot_time_range = plot_time_range
        assert plot_time_range is None or plot_time_range[1] > plot_time_range[0], "Invalid time range!"
        # Initialize empty lists for each variable in plot_attributes
        for var in self.plt_attr:
            self.plt_data[var] = []

        # assert time is always a key in the plot_attributes
        assert "time" in self.plt_attr, "Time variable not in plot_attributes!"

        self.color_dict = {
            "white"       : (255, 255, 255), 
            "green"       : (  0, 255,   0), 
            "red"         : (  0,   0, 255), 
            "light_blue"  : (255, 165,   0)
            }
        self.colors = [color for color, gbr in self.color_dict.items()]

    def record_plot_data(self, sim: MjcSim) -> None:
        """Log specified variables."""
        # self.plt_data['time'].append(sim.data.time) # Always log time

        for var in self.plt_attr:
            value = sim.data.time if var == "time" else getattr(sim, var, None)
            assert value is not None, f"Variable {var} not found."
            self.plt_data[var].append(value)

    def record_frame(self, sim: MjcSim) -> None:
        """Record robot frames at specified intervals."""
        if sim.data.time >= self.next_vid_output_time:
            self.robot_frames.append(sim.get_image())
            self.next_vid_output_time += self.output_interval

    # def export_clip(self, 
    #                 frames: list, 
    #                 vid_type: str, 
    #                 output_path: str) -> None:
    #     """Export video from frames."""
    #     clip = ImageSequenceClip(frames, fps=self.video_fps)
    #     from proglog import TqdmProgressBarLogger
    #     custom_logger = TqdmProgressBarLogger(logged_bars='all', print_messages=False)
    #     clip.write_videofile(output_path, 
    #                          codec="libx264", 
    #                          preset="medium",
    #                          logger=custom_logger)
    #     print(f"{vid_type} Video saved to {output_path}")

    def export_clip(self,
                frames: list,
                vid_type: str,
                output_path: str) -> None:
        """Export video using FFmpeg subprocess for reliable libx264 encoding."""
        import subprocess
        import tempfile
        import cv2
        import os
        from tqdm import tqdm
        
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames as JPG files (much faster than PNG)
            for i, frame in enumerate(tqdm(frames, desc=f"Saving frames for {vid_type}")):
                # Ensure BGR format for OpenCV
                if frame.shape[2] == 3:  # RGB
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                # Save frame
                cv2.imwrite(os.path.join(temp_dir, f"frame_{i:05d}.jpg"), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Direct FFmpeg call for libx264 encoding with reduced verbosity
            subprocess.run([
                'ffmpeg', '-y',
                '-framerate', str(self.video_fps),
                '-i', os.path.join(temp_dir, 'frame_%05d.jpg'),
                '-c:v', 'libx264',
                '-preset', 'medium',  # Can use 'ultrafast' for speed at quality cost
                '-pix_fmt', 'yuv420p',
                '-loglevel', 'warning',  # Reduce FFmpeg output verbosity
                output_path
            ], check=True)
            
        print(f"{vid_type} Video saved to {output_path}")


    def generate_robot_video(self, output_path: str) -> str:
        """Create a video from stored robot frames."""
        assert self.robot_frames is not [], "No frames recorded!"
        self.export_clip(self.robot_frames, "Scene", output_path)
        return output_path

    def setup_plot(self) -> None:
        """Setup the plot format for the recorded data."""
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Live Plot Video")

        plot_w = 540 # Width of the plot
        plot_hs = [] # Heights of the plots

        self.plots_and_curves = []
        # Iterate over plot structure to create subplots
        for row, subplot_vars in enumerate(self.plt_struc):
            x = subplot_vars[0]  # First variable is the x-axis
            ys = subplot_vars[1:]  # Remaining variables are y-axes
            plot_title_list = [self.plt_attr[y]['title'] for y in ys]
            plot = self.win.addPlot(row=row, col=0, title=", ".join(plot_title_list)) # Plots stacked vertically
        
            # Create appropriate x/y labels for the plot
            first_y_unit = self.plt_attr[ys[0]]['unit']
            if len(ys) == 1:
                ylabel = (f"{self.plt_attr[ys[0]]['title']}"
                          f"({self.plt_attr[ys[0]]['unit']})")
            elif all(self.plt_attr[y]['unit'] == first_y_unit for y in ys):
                ylabel = (f"Values ({self.plt_attr[ys[0]]['unit']})")

            xlabel = (f"{self.plt_attr[x]['title']}"
                      f"({self.plt_attr[x]['unit']})")
            plot.setLabels(left=ylabel, bottom=xlabel)
            plot.showGrid(x=True, y=True, alpha=0.5)
            plot.addLegend(offset=(10, 10))

            # Square aspect ratio for non-time plots, else 2:1
            plot_h = plot_w // 2 if self.plt_attr[x]['unit'] == 's' else plot_w
            plot.setPreferredSize(plot_w, plot_h)
            plot_hs.append(plot_h)

            curves = []
            for idx, y in enumerate(ys):
                attrs = self.plt_attr[y]      
                curve_color = self.color_dict[self.colors[idx % len(self.colors)]] # Cycle through colors          
                curve = plot.plot(pen=pg.mkPen(color=curve_color, width=2), name=attrs['title'])
                curves.append((curve, y))
            self.plots_and_curves.append((plot, curves, x))

    
        for (plot, curves, x) in self.plots_and_curves:
            xmin, xmax = min(self.plt_data[x]), max(self.plt_data[x])
            ymin = min(min(self.plt_data[y]) for _, y in curves)
            ymax = max(max(self.plt_data[y]) for _, y in curves)
            plot.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax))
            if self.plot_time_range is not None:
                plot.setRange(xRange=self.plot_time_range)

        self.win.resize(plot_w, sum(plot_hs))
        self.win.show()
        self.app.processEvents()

    def generate_plot_video(self, output_path: str) -> str:
        """Create a video from stored plot data."""
        self.setup_plot()
        self.plot_frames = []
        next_plot_output_time = 0

        for i, t in enumerate(self.plt_data['time']):
            if t < next_plot_output_time: continue
            
            for (_, curves, x) in self.plots_and_curves:
                x_data = self.plt_data[x][:i + 1]
                for curve, y in curves:
                    curve.setData(x_data, self.plt_data[y][:i + 1])

            img = self.win.grab().toImage()
            buffer = np.array(img.bits().asarray(img.byteCount()))
            buffer.shape = (img.height(), img.width(), 4)
            self.plot_frames.append(buffer[:, :, :3])
            next_plot_output_time += self.output_interval

        self.win.close()
        self.export_clip(self.plot_frames, "Plot", output_path)
        return output_path

    def stack_video_frames(self, 
                           left_frames: list, 
                           right_frames: list, 
                           output_path: str, 
                           pad: bool=True):
        """Stack two lists of frames side by side, using either padding (default) or resizing."""
        # only padding right now, stretching keeps killing the process due to excessive memory usage
        def adjust_height(frame, target_h, pad_mode):
            """Helper to adjust frame height using either padding or resizing."""
            h, w, _ = frame.shape
            if h == target_h:
                return frame
            if pad_mode:
                pad_width = (target_h - h) // 2
                return np.pad(frame, ((pad_width, pad_width), (0, 0), (0, 0)))
            return cv2.resize(frame, (w * target_h // h, target_h))  # Preserve aspect ratio

        target_h = max(left_frames[0].shape[0], right_frames[0].shape[0])  # Determine max height
        combined_frames = [cv2.hconcat([adjust_height(left, target_h, pad),
                                        adjust_height(right, target_h, pad)])
                                        for left, right in zip(left_frames, right_frames)]
        self.export_clip(combined_frames, "Combined", output_path)