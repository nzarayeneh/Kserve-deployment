import torch
import kserve
from google.cloud import storage
from kserve import Model, Storage
from tempfile import TemporaryFile
from kserve.model import ModelMissingError, InferenceError
from typing import Dict
import logging
import pyTigerGraph as tg
from torch_geometric.nn import GCN
import os 

logger = logging.getLogger(__name__)
conn = tg.TigerGraphConnection("http://35.230.92.92", graphname="Cora")

# Hyperparameters
hp = {"batch_size": 64, "num_neighbors": 10, "num_hops": 2, "hidden_dim": 64,
      "num_layers": 2, "dropout": 0.6, "lr": 0.01, "l2_penalty": 5e-4}

class GCNNodeClassifier(Model):
    def __init__(self, name: str, bucket_name:str='tg_models', model_bucket:str='model.state'):
        super().__init__(name)
        self.name = name
        self.bucket_name = bucket_name
        self.model_bucket = model_bucket
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.infer_loader = conn.gds.neighborLoader(
            v_in_feats=["x"],
            v_out_labels=["y"],
            output_format="PyG",
            batch_size=hp["batch_size"],
            num_neighbors=hp["num_neighbors"],
            num_hops=hp["num_hops"],
            shuffle=False
        )
        #load the model from gstorage
        try:
            self.model = self.load_model()
        except ModelMissingError:
            logging.error(f"fail to locate model file for model {model_name} under the bucket name {bucket_name},"
                      f"trying loading from model repository.")

    def load(self):
        pass
    
    def load_model(self):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)
        #select bucket file
        blob = bucket.blob(self.model_bucket)
        with TemporaryFile() as temp_file:
            #download blob into temp file
            blob.download_to_file(temp_file)
            temp_file.seek(0)
            model = GCN(
                in_channels=1433,
                hidden_channels=hp["hidden_dim"],
                num_layers=hp["num_layers"],
                out_channels=7,
                dropout=hp["dropout"],
            )
            logger.info("Instantiated Model")
            model.load_state_dict(torch.load(temp_file))
            model.to(self.device).eval()
            logger.info("Loaded Model")
        return model

    def predict(self, request: Dict) -> Dict:
        input_nodes = request["nodes"]
        input_ids = set([str(node['primary_id']) for node in input_nodes])
        logger.info(input_ids)
        data = self.infer_loader.fetch(input_nodes)
        logger.info (f"predicting {data}")
        pred = self.model(data.x.float(), data.edge_index).argmax(dim=1)
        ret = {"predictions": []}
        for primary_id, label in zip(data.primary_id, pred):
            if primary_id in input_ids:
                ret['predictions'].append({'primary_id': primary_id, 'label': label.item()})
        return ret

if __name__ == "__main__":
    model_name = os.environ.get('K_SERVICE', "tg-gcn-kserve-demo-predictor-default")
    model_name = '-'.join(model_name.split('-')[:-2]) # removing suffix "-predictor-default"
    print(model_name)
    logging.info(f"Starting model '{model_name}'")
    model = GCNNodeClassifier(model_name)
    kserve.ModelServer(http_port=8080).start([model])
