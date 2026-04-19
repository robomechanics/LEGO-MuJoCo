import argparse

def arg_parser(name) -> dict:
    """Parse command-line arguments for the named simulation."""
    parser = argparse.ArgumentParser(description=f"{name} sim")
    parser.add_argument("-n", "--name", type=str, default=f"{name}_sim_default", help="name of the video")
    parser.add_argument("-t", "--sim_time", type=float, default=5, help="total simulation time")
    parser.add_argument("-com", "--com", action="store_true",
                        help="export the XY center-of-mass trajectory plot after the simulation")
    parser.add_argument("-gui", "--gui", action="store_true", help="enable GUI")
    parser.add_argument("-r", "--record", action="store_true", help="record video")
    parser.add_argument("-rdir", "--robot_dir", type=str, default="robots", help="directory of robots")
    parser.add_argument("-vdir", "--video_dir", type=str, default="data/videos", help="directory of videos")
    parser.add_argument("-vfps", "--video_fps", type=int, default=30, help="FPS for video recording")
    
    return vars(parser.parse_args())  # Convert to dictionary
