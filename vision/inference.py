class InferenceEngine:

    def __init__(

        self,

        runtime_model,

    ):

        self.predictor = Predictor(runtime_model)

        self.converter = DetectionConverter()

    ############################################################

    def infer(self, image):

        results = self.predictor.predict(image)

        detections = self.converter.convert(results)

        return detections