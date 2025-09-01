import subprocess


class COLMAPFrame:
    default_conf = {
        "default": "matches",
        "summary_visible": False,
    }

    def __init__(self, conf, data, preds, title=None, event=1, summaries=None):
        self.conf = conf
        self.data = data
        self.preds = preds
        self.names = list(preds.keys())
        self.summaries = summaries
        self.gui_processes = []
        self.show()

    def show(self):
        print("Opening COLMAP GUI for each benchmark... ")
        for benchmark, pred in self.preds.items():
            output_dir = pred["output_dir"][0]
            cmd = [
                "colmap",
                "gui",
                "--image_path",
                self.data["reconstruction"][0].image_dir,
                "--database_path",
                output_dir + "database.db",
                "--import_path",
                output_dir,
            ]
            self.gui_processes.append(subprocess.Popen(cmd))

    def close(self):
        for process in self.gui_processes:
            process.terminate()
        self.gui_processes = []
