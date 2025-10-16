import os
# Utils file

# IO stores all the file paths that need to be accessed or written to.
class IO():
    def __init__(self):
        self.transition_output_path = "output/Transition_Images/"
        self.transition_output_file_name = "nbst_"
        self.output_image_path = "output/Final_Image/"
        self.output_gif_path = "output/Animation/"
        self.output_file_name = "Style_Transfer.jpg"
        self.intermediate_images_file_name = "Intermediate_Images.jpg"
        self.gif_file_name = "nbst_animation.gif"
        self.mask_image_path = "style_trans/util_images/mask.jpg"
        self.util_images_path = "style_trans/util_images"
        self._make_dirs_()
        
    def _make_dirs_(self):
        if not os.path.exists(self.util_images_path):
            os.makedirs(self.util_images_path)
        if not os.path.exists(self.output_gif_path):
            os.makedirs(self.output_gif_path)
        if not os.path.exists(self.output_image_path):
            os.makedirs(self.output_image_path)
        if not os.path.exists(self.transition_output_path):
            os.makedirs(self.transition_output_path)
