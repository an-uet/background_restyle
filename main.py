from style_trans.naive_background_style_transfer import NaiveBackgroundStyleTransfer
if __name__ == '__main__':

    content = "input_images/Content/portrait.jpg"
    style = "input_images/Style/Starry_Night.jpg"
    nbst = NaiveBackgroundStyleTransfer(number_of_epochs=100, verbose = True)
    nbst.perform(content, style)
    nbst.generate_gif(speed=1.5)
