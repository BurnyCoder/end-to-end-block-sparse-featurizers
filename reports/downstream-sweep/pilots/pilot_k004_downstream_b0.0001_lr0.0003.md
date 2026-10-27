# Pilot: pilot_k004_downstream_b0.0001_lr0.0003

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k004_downstream_b0.0001_lr0.0003",
  "cell": "k004_downstream_b0.0001",
  "label": "KL + 0.0001 downstream MSE",
  "k": 4,
  "lr": 0.0003,
  "seed": 900,
  "state": {
    "cell": {
      "k": 4,
      "objective": "downstream",
      "beta": 0.0001
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "complete",
    "elapsed_seconds": 68.0,
    "train_seconds": 64.75099999998929,
    "dev_seconds": 2.9679999999934807,
    "peak_vram_bytes": 1199974912,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": {
    "kl": 2.6021931171417236,
    "ce_delta": 2.6072452068328857,
    "accuracy": 0.2240000069141388,
    "teacher_accuracy": 0.9919999837875366,
    "agreement": 0.2199999988079071,
    "images": 250,
    "layer_mse": [
      [
        1.6234924793243408,
        1.2986807823181152,
        14.356094360351562
      ],
      [
        5.633850574493408,
        5.395187854766846,
        14.989542007446289
      ],
      [
        11.678555488586426,
        11.489505767822266,
        19.089475631713867
      ],
      [
        30.028085708618164,
        30.174999237060547,
        24.269500732421875
      ],
      [
        78.50117492675781,
        79.23863220214844,
        49.591514587402344
      ],
      [
        235.1569366455078,
        237.2497100830078,
        153.11822509765625
      ],
      [
        1483.05908203125,
        1492.101806640625,
        1128.584716796875
      ]
    ],
    "layer_nmse": [
      [
        0.901333212852478,
        0.9128664135932922,
        0.862684428691864
      ],
      [
        1.6373236179351807,
        1.7443346977233887,
        0.8776487708091736
      ],
      [
        1.4948099851608276,
        1.5242959260940552,
        1.0263855457305908
      ],
      [
        1.5143688917160034,
        1.5248554944992065,
        1.134211540222168
      ],
      [
        1.6154266595840454,
        1.6165306568145752,
        1.549127221107483
      ],
      [
        1.5402830839157104,
        1.5337433815002441,
        2.0785937309265137
      ],
      [
        2.205829620361328,
        2.1939427852630615,
        3.0670552253723145
      ]
    ],
    "insertion_nmse": 0.9013331532478333,
    "downstream_mse": 307.34295654296875,
    "epoch": 1,
    "images_seen": 510
  },
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 3.407701790332794,
      "kl": 3.3756935000419617,
      "insertion_nmse": 0.9228818565607071,
      "downstream_mse": 320.08287048339844,
      "lr": 0.0002974060128544361,
      "gradient_norm": 5.058083534240723,
      "elapsed_seconds": 4.062000000005355
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 3.506369948387146,
      "kl": 3.4745545983314514,
      "insertion_nmse": 0.832123801112175,
      "downstream_mse": 318.1538543701172,
      "lr": 0.0002897237368890237,
      "gradient_norm": 5.185628414154053,
      "elapsed_seconds": 7.953000000008615
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 3.4450785517692566,
      "kl": 3.413409560918808,
      "insertion_nmse": 0.9303913861513138,
      "downstream_mse": 316.6897659301758,
      "lr": 0.0002772483976608436,
      "gradient_norm": 5.194047927856445,
      "elapsed_seconds": 11.953000000008615
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 3.2041686177253723,
      "kl": 3.172438383102417,
      "insertion_nmse": 0.8540538549423218,
      "downstream_mse": 317.3025588989258,
      "lr": 0.0002604594154601839,
      "gradient_norm": 5.231679439544678,
      "elapsed_seconds": 16.125
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 4.039505660533905,
      "kl": 4.00754451751709,
      "insertion_nmse": 0.9234521985054016,
      "downstream_mse": 319.6126480102539,
      "lr": 0.0002400019814576463,
      "gradient_norm": 5.438055992126465,
      "elapsed_seconds": 20.171999999991385
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 3.4074110984802246,
      "kl": 3.3761746287345886,
      "insertion_nmse": 0.8329566270112991,
      "downstream_mse": 312.3647232055664,
      "lr": 0.00021666226336928708,
      "gradient_norm": 5.004032611846924,
      "elapsed_seconds": 24.219000000011874
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 2.848561644554138,
      "kl": 2.8170864284038544,
      "insertion_nmse": 0.9306478202342987,
      "downstream_mse": 314.7523651123047,
      "lr": 0.00019133719347217733,
      "gradient_norm": 4.848941326141357,
      "elapsed_seconds": 28.328000000008615
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 2.928545832633972,
      "kl": 2.8980615735054016,
      "insertion_nmse": 0.872948169708252,
      "downstream_mse": 304.8422393798828,
      "lr": 0.000165,
      "gradient_norm": 5.6164422035217285,
      "elapsed_seconds": 32.40600000001723
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 3.6083186268806458,
      "kl": 3.577441096305847,
      "insertion_nmse": 0.7745058685541153,
      "downstream_mse": 308.7758026123047,
      "lr": 0.00013866280652782267,
      "gradient_norm": 5.626955509185791,
      "elapsed_seconds": 36.671999999991385
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 2.751846969127655,
      "kl": 2.7204484343528748,
      "insertion_nmse": 0.8442890644073486,
      "downstream_mse": 313.9850769042969,
      "lr": 0.00011333773663071288,
      "gradient_norm": 5.49575138092041,
      "elapsed_seconds": 40.76600000000326
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 2.7365442514419556,
      "kl": 2.705837607383728,
      "insertion_nmse": 0.8609178364276886,
      "downstream_mse": 307.06678771972656,
      "lr": 8.999801854235373e-05,
      "gradient_norm": 4.874447345733643,
      "elapsed_seconds": 44.828000000008615
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 2.714705467224121,
      "kl": 2.6832703351974487,
      "insertion_nmse": 0.8743550479412079,
      "downstream_mse": 314.3511505126953,
      "lr": 6.954058453981609e-05,
      "gradient_norm": 5.643758773803711,
      "elapsed_seconds": 48.875
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 2.602059066295624,
      "kl": 2.5704153776168823,
      "insertion_nmse": 0.8108852207660675,
      "downstream_mse": 316.43704986572266,
      "lr": 5.275160233915637e-05,
      "gradient_norm": 5.078738689422607,
      "elapsed_seconds": 52.875
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 2.7830464243888855,
      "kl": 2.752394437789917,
      "insertion_nmse": 0.8736593574285507,
      "downstream_mse": 306.52003479003906,
      "lr": 4.027626311097629e-05,
      "gradient_norm": 5.345433712005615,
      "elapsed_seconds": 56.844000000011874
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 2.6924136877059937,
      "kl": 2.662024676799774,
      "insertion_nmse": 0.8990781009197235,
      "downstream_mse": 303.89070892333984,
      "lr": 3.259398714556389e-05,
      "gradient_norm": 5.408639907836914,
      "elapsed_seconds": 60.796999999991385
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 3.0775601387023928,
      "kl": 3.0461645285288492,
      "insertion_nmse": 0.9720046162605286,
      "downstream_mse": 313.9559387207031,
      "lr": 2.9999999999999997e-05,
      "gradient_norm": 6.347273349761963,
      "elapsed_seconds": 64.84400000001187
    }
  ]
}
```
