import argparse



class Args(object):
    """
    设置命令行参数的接口
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def set_main_args(self):
        """options for train"""
        self.parser.add_argument("--video_path", type=str, default="../../video/202-233/FormatFactoryPart3.mp4")
        self.parser.add_argument("--nowframe", type=int, default=1, help="start frame")
        self.parser.add_argument("--dataset_dir", type=str, default="../../bigcoal/202-233-part3")
        self.parser.add_argument("--qSize", type=int, default=2)
        self.parser.add_argument("--processFrame", type=int, default=3000)
        self.args = self.parser.parse_args()

