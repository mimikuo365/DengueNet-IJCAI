import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input, Model
from keras.layers import Permute, Dense, concatenate
import keras_tuner
from transformers import TFViTModel, ViTFeatureExtractor


class DengueNet(keras_tuner.HyperModel):
    unit_ls = [60, 30, 5]
    nn_unit_ls = [30, 1]
    learning_rate = 0.001
    trainable_layer = 1
    vit_backbone = "google/vit-base-patch16-224-in21k"
    leaky = 0.3
    vit_trainable = False

    models = {
        "Vit": 1,  # RGB images
        "Feng": 2,  # Radiomics
        "Mobile": 3,  # RGB images
        "Case": 4,  # Cases
        "FengMobileVit": 5,  # Radiomics   # RGB images   # RGB images
        "MobileVit": 6,  # RGB images  # RGB images
        "FengMobile": 7,  # Radiomics   # RGB images
        "FengVit": 8,  # Radiomics   # RGB images
        "Ensemble": 9,  # Radiomics   # RGB images   # RGB images   # Cases
        "VitFengCase": 10,  # RGB images, # Radiomics, # Cases
    }

    @staticmethod
    def getHyperparameter(
        leaky_rate=None, vit_trainable=None, learning_rate=None
    ) -> str:
        name = ""
        if leaky_rate:
            DengueNet.leaky = leaky_rate
        if vit_trainable:
            DengueNet.vit_trainable = vit_trainable
        if learning_rate:
            DengueNet.learning_rate = learning_rate
        print(DengueNet.leaky, leaky_rate, DengueNet.vit_trainable, vit_trainable)
        for i, unit in enumerate(DengueNet.unit_ls):
            name += str(unit) + "_"
        for i, unit in enumerate(DengueNet.nn_unit_ls):
            name += str(unit) + "_"
        return f"{name}{DengueNet.trainable_layer}_{DengueNet.learning_rate}_{DengueNet.leaky}_{DengueNet.vit_trainable}"

    @staticmethod
    def getCnnModel(include_top, img_shape):
        model = tf.keras.applications.MobileNetV3Small(
            include_top=include_top, weights="imagenet", input_shape=img_shape
        )
        for idx, layer in enumerate(model.layers):
            layer.trainable = idx > len(model.layers) - DengueNet.trainable_layer
        return model

    @staticmethod
    def createLstmLayers(layer):
        for i, unit in enumerate(DengueNet.unit_ls):
            layer = layers.LSTM(
                units=unit,
                dropout=0.1,
                return_sequences=(i != len(DengueNet.unit_ls) - 1),
            )(layer)
        return layer

    @staticmethod
    def getVitBackbone(target_size):
        feature_extractor = ViTFeatureExtractor.from_pretrained(DengueNet.vit_backbone)
        model = TFViTModel.from_pretrained(DengueNet.vit_backbone)
        if DengueNet.vit_trainable:
            model.vit.embeddings.trainable = True
            model.vit.encoder.trainable = False
            model.vit.layernorm.trainable = False
            model.vit.pooler.trainable = False

        else:
            model.trainable = False

        inputs = Input(target_size)
        channel_fist_inputs = Permute((3, 1, 2))(inputs)
        embeddings = model.vit(channel_fist_inputs)[0][:, 0, :]
        vit_model = Model(inputs=inputs, outputs=embeddings)
        return vit_model

    def __init__(self, setting, leaky_rate):
        self.setting = setting
        self.model = setting["model"]
        DengueNet.leaky_rate = leaky_rate

    def getVitModel(self):
        lstm_week = self.setting["lstm_weeks"]
        img_shape = self.setting["resized_img_shape"]
        vit_input = Input(shape=np.concatenate((lstm_week, img_shape), axis=None))
        vit_model = DengueNet.getVitBackbone(img_shape)
        layer = layers.TimeDistributed(
            vit_model, input_shape=((lstm_week,) + img_shape)
        )(vit_input)
        vit_layer = DengueNet.createLstmLayers(layer)
        return vit_layer, vit_input

    def getMobileNetModel(self):
        lstm_week = self.setting["lstm_weeks"]
        img_shape = self.setting["resized_img_shape"]
        include_top = self.setting["include_top"]
        cnn_input = Input(shape=np.concatenate((lstm_week, img_shape), axis=None))
        cnn_model = DengueNet.getCnnModel(include_top, img_shape)
        layer = layers.TimeDistributed(
            cnn_model, input_shape=((lstm_week,) + img_shape)
        )(cnn_input)
        layer = layers.Reshape((layer.shape[1], -1))(layer)
        cnn_layer = DengueNet.createLstmLayers(layer)
        return cnn_layer, cnn_input

    def getFengModel(self):
        lstm_week = self.setting["lstm_weeks"]
        feature_per_sample = self.setting["len_feature"]
        eng_input = Input(shape=(lstm_week, feature_per_sample))
        eng_layer = DengueNet.createLstmLayers(eng_input)
        return eng_layer, eng_input

    def getCaseModel(self):
        lstm_week = self.setting["lstm_weeks"]
        case_input = Input(shape=(lstm_week, 1))
        case_layer = DengueNet.createLstmLayers(case_input)
        return case_layer, case_input

    def getConcatModel(
        self, layer_ls, input_ls, learning_rate=None, use_leaky=True, leaky_rate=None
    ):
        if not leaky_rate:
            leaky_rate = DengueNet.leaky
        layer = concatenate(layer_ls, name="concatenated_layer")
        if use_leaky:
            for unit in DengueNet.nn_unit_ls:
                layer = Dense(
                    units=unit, activation=tf.keras.layers.LeakyReLU(alpha=leaky_rate)
                )(layer)
        else:
            for unit in DengueNet.nn_unit_ls:
                layer = Dense(units=unit, activation="relu")(layer)

        model = Model(inputs=[input_ls], outputs=[layer], name="merged_model")
        if learning_rate is None:
            learning_rate = DengueNet.learning_rate
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            loss="mean_absolute_error", optimizer=opt, metrics=["mean_absolute_error"]
        )
        return model

    def build(self, hp):
        params = {
            "learning_rate": hp.Float(
                "learning_rate", min_value=0.0001, max_value=0.01, sampling="log"
            ),
        }
        model_idx = self.models[self.model]
        if model_idx == 1:
            return self.createVit(params["learning_rate"])
        elif model_idx == 2:
            return self.createFeatureEng(params["learning_rate"])
        elif model_idx == 3:
            return self.createMobileNet(params["learning_rate"])
        elif model_idx == 4:
            return self.createCaseModel(params["learning_rate"])
        elif model_idx == 5:
            return self.createFengMobileVit(params["learning_rate"])
        elif model_idx == 6:
            return self.createMobileVit(params["learning_rate"])
        elif model_idx == 7:
            return self.createMobileFeng(params["learning_rate"])
        elif model_idx == 8:
            return self.createFengVit(params["learning_rate"])
        elif model_idx == 9:
            return self.createEnsemble(params["learning_rate"])
        else:
            print(f"Error: Not existed model {self.model}")

    def create(self):
        model_idx = self.models[self.model]
        if model_idx == 1:
            return self.createVit()
        elif model_idx == 2:
            return self.createFeatureEng()
        elif model_idx == 3:
            return self.createMobileNet()
        elif model_idx == 4:
            return self.createCaseModel()
        elif model_idx == 5:
            return self.createFengMobileVit()
        elif model_idx == 6:
            return self.createMobileVit()
        elif model_idx == 7:
            return self.createMobileFeng()
        elif model_idx == 8:
            return self.createFengVit()
        elif model_idx == 9:
            return self.createEnsemble()
        elif model_idx == 10:
            return self.createVitFengCase()
        else:
            print(f"Error: Not existed model {self.model}")

    def createFengMobileVit(self, learning_rate=None):
        eng_layer, eng_input = self.getFengModel()
        cnn_layer, cnn_input = self.getMobileNetModel()
        vit_layer, vit_input = self.getVitModel()
        layer_ls = [eng_layer, cnn_layer, vit_layer]
        input_ls = [eng_input, cnn_input, vit_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model

    def createFengVit(self, learning_rate=None):
        eng_layer, eng_input = self.getFengModel()
        vit_layer, vit_input = self.getVitModel()
        layer_ls = [eng_layer, vit_layer]
        input_ls = [eng_input, vit_input]
        model = self.getConcatModel(
            layer_ls, input_ls, learning_rate, use_leaky=False, leaky_rate=0.2
        )
        return model

    def createMobileVit(self, learning_rate=None):
        cnn_layer, cnn_input = self.getMobileNetModel()
        vit_layer, vit_input = self.getVitModel()
        layer_ls = [vit_layer, cnn_layer]
        input_ls = [vit_input, cnn_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model

    def createMobileFeng(self, learning_rate=None):
        eng_layer, eng_input = self.getFengModel()
        cnn_layer, cnn_input = self.getMobileNetModel()
        layer_ls = [eng_layer, cnn_layer]
        input_ls = [eng_input, cnn_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model

    def createFeatureEng(self, learning_rate=None):
        eng_layer, eng_input = self.getFengModel()
        layer_ls = [eng_layer]
        input_ls = [eng_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model

    def createVit(self, learning_rate=None):
        vit_layer, vit_input = self.getVitModel()
        layer_ls = [vit_layer]
        input_ls = [vit_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model

    def createMobileNet(self, learning_rate=None):
        cnn_layer, cnn_input = self.getMobileNetModel()
        layer_ls = [cnn_layer]
        input_ls = [cnn_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model

    def createEnsemble(self, learning_rate=None):
        eng_layer, eng_input = self.getFengModel()
        cnn_layer, cnn_input = self.getMobileNetModel()
        vit_layer, vit_input = self.getVitModel()
        case_layer, case_input = self.getCaseModel()
        layer_ls = [eng_layer, cnn_layer, vit_layer, case_layer]
        input_ls = [eng_input, cnn_input, vit_input, case_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model

    def createVitFengCase(self, learning_rate=None):
        eng_layer, eng_input = self.getFengModel()
        vit_layer, vit_input = self.getVitModel()
        case_layer, case_input = self.getCaseModel()
        layer_ls = [eng_layer, vit_layer, case_layer]
        input_ls = [eng_input, vit_input, case_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model

    def createCaseModel(self, learning_rate=None):
        case_layer, case_input = self.getCaseModel()
        layer_ls = [case_layer]
        input_ls = [case_input]
        model = self.getConcatModel(layer_ls, input_ls, learning_rate)
        return model
