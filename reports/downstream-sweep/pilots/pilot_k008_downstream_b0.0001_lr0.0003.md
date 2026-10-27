# Pilot: pilot_k008_downstream_b0.0001_lr0.0003

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k008_downstream_b0.0001_lr0.0003",
  "cell": "k008_downstream_b0.0001",
  "label": "KL + 0.0001 downstream MSE",
  "k": 8,
  "lr": 0.0003,
  "seed": 900,
  "state": {
    "cell": {
      "k": 8,
      "objective": "downstream",
      "beta": 0.0001
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "trained_epoch",
    "elapsed_seconds": 57.23399999999674,
    "train_seconds": 57.10899999999674,
    "dev_seconds": 0.0,
    "peak_vram_bytes": 1199974912,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": null,
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 3.153047025203705,
      "kl": 3.1252062916755676,
      "insertion_nmse": 0.8621038347482681,
      "downstream_mse": 278.4066467285156,
      "lr": 0.0002974060128544361,
      "gradient_norm": 8.631329536437988,
      "elapsed_seconds": 3.6559999999881256
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 2.9059163331985474,
      "kl": 2.8781617879867554,
      "insertion_nmse": 0.777438759803772,
      "downstream_mse": 277.5458450317383,
      "lr": 0.0002897237368890237,
      "gradient_norm": 9.340599060058594,
      "elapsed_seconds": 7.217999999993481
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 2.479110896587372,
      "kl": 2.451886236667633,
      "insertion_nmse": 0.8688755482435226,
      "downstream_mse": 272.24698638916016,
      "lr": 0.0002772483976608436,
      "gradient_norm": 7.433170795440674,
      "elapsed_seconds": 10.796000000002095
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 2.644787847995758,
      "kl": 2.616946578025818,
      "insertion_nmse": 0.7976184040307999,
      "downstream_mse": 278.4125061035156,
      "lr": 0.0002604594154601839,
      "gradient_norm": 8.152755737304688,
      "elapsed_seconds": 14.35899999999674
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 3.163430094718933,
      "kl": 3.135877549648285,
      "insertion_nmse": 0.8639301210641861,
      "downstream_mse": 275.5255584716797,
      "lr": 0.0002400019814576463,
      "gradient_norm": 7.934195041656494,
      "elapsed_seconds": 17.921000000002095
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 2.898604393005371,
      "kl": 2.8711673617362976,
      "insertion_nmse": 0.7789728194475174,
      "downstream_mse": 274.3703308105469,
      "lr": 0.00021666226336928708,
      "gradient_norm": 7.798094272613525,
      "elapsed_seconds": 21.51500000001397
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 2.7254398465156555,
      "kl": 2.6977020502090454,
      "insertion_nmse": 0.8688274919986725,
      "downstream_mse": 277.3779830932617,
      "lr": 0.00019133719347217733,
      "gradient_norm": 7.625788688659668,
      "elapsed_seconds": 25.09299999999348
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 2.1957393884658813,
      "kl": 2.1690618991851807,
      "insertion_nmse": 0.8160014152526855,
      "downstream_mse": 266.7752685546875,
      "lr": 0.000165,
      "gradient_norm": 7.75779914855957,
      "elapsed_seconds": 28.75
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 2.969372808933258,
      "kl": 2.942400813102722,
      "insertion_nmse": 0.7237977534532547,
      "downstream_mse": 269.71961212158203,
      "lr": 0.00013866280652782267,
      "gradient_norm": 8.184257507324219,
      "elapsed_seconds": 32.375
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 2.469227761030197,
      "kl": 2.4414773285388947,
      "insertion_nmse": 0.788795530796051,
      "downstream_mse": 277.5047912597656,
      "lr": 0.00011333773663071288,
      "gradient_norm": 8.490187644958496,
      "elapsed_seconds": 35.905999999988126
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 2.5055713653564453,
      "kl": 2.4786306619644165,
      "insertion_nmse": 0.8046276420354843,
      "downstream_mse": 269.4069595336914,
      "lr": 8.999801854235373e-05,
      "gradient_norm": 6.9783549308776855,
      "elapsed_seconds": 39.5
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 2.2032168209552765,
      "kl": 2.175276458263397,
      "insertion_nmse": 0.8182318955659866,
      "downstream_mse": 279.40370178222656,
      "lr": 6.954058453981609e-05,
      "gradient_norm": 7.097571849822998,
      "elapsed_seconds": 42.98399999999674
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 2.349169969558716,
      "kl": 2.321086138486862,
      "insertion_nmse": 0.7584886848926544,
      "downstream_mse": 280.8387756347656,
      "lr": 5.275160233915637e-05,
      "gradient_norm": 7.096966743469238,
      "elapsed_seconds": 46.625
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 2.431984543800354,
      "kl": 2.4047319293022156,
      "insertion_nmse": 0.816657155752182,
      "downstream_mse": 272.5262451171875,
      "lr": 4.027626311097629e-05,
      "gradient_norm": 7.345781326293945,
      "elapsed_seconds": 50.14000000001397
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 2.249593734741211,
      "kl": 2.2226862609386444,
      "insertion_nmse": 0.8400874137878418,
      "downstream_mse": 269.07437896728516,
      "lr": 3.259398714556389e-05,
      "gradient_norm": 6.992637634277344,
      "elapsed_seconds": 53.671000000002095
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 2.663457425435384,
      "kl": 2.635564120610555,
      "insertion_nmse": 0.9095206379890441,
      "downstream_mse": 278.9334981282552,
      "lr": 2.9999999999999997e-05,
      "gradient_norm": 9.238889694213867,
      "elapsed_seconds": 57.23399999999674
    }
  ]
}
```
