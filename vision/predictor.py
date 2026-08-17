from vision.postprocessing.preprocessing import ImagePreprocessor


class Predictor:

    def __init__(self, runtime_model):

        self.runtime_model = runtime_model

        self.preprocessor = ImagePreprocessor()

    ############################################################

    def predict(

        self,

        image,

        confidence=0.25,

        iou=0.45,

    ):

        image = self.preprocessor.preprocess(image)

        return self.runtime_model.model(

            image,

            conf=confidence,

            iou=iou,

            verbose=False,

        )