from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import LOGGER, ROOT, is_git_dir
from ultralytics.yolo.v8.classify.predict import ClassificationPredictor

class classify_sam(YOLO):

    def predict_classification(self, source=None, **kwargs):
        """
        Perform classification prediction using the YOLO classification model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            **kwargs : Additional keyword arguments passed to the classification predictor.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The classification prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        overrides = self.overrides.copy()
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = 'classify'
        overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        self.classification_predictor = ClassificationPredictor(overrides=overrides)
        self.classification_predictor.setup_model(model=self.model, verbose=False)
        try:
            return self.classification_predictor(source)
        except Exception as e:
            return None
