# Pilot: pilot_k004_downstream_b0.001_lr0.0003

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k004_downstream_b0.001_lr0.0003",
  "cell": "k004_downstream_b0.001",
  "label": "KL + 0.001 downstream MSE",
  "k": 4,
  "lr": 0.0003,
  "seed": 900,
  "state": {
    "cell": {
      "k": 4,
      "objective": "downstream",
      "beta": 0.001
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "complete",
    "elapsed_seconds": 61.421999999991385,
    "train_seconds": 58.609000000025844,
    "dev_seconds": 2.547000000020489,
    "peak_vram_bytes": 1199974912,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": {
    "kl": 2.5875449180603027,
    "ce_delta": 2.591388702392578,
    "accuracy": 0.2199999988079071,
    "teacher_accuracy": 0.9919999837875366,
    "agreement": 0.2160000056028366,
    "images": 250,
    "layer_mse": [
      [
        1.6233092546463013,
        1.298622965812683,
        14.350993156433105
      ],
      [
        5.628937721252441,
        5.390268325805664,
        14.984748840332031
      ],
      [
        11.678720474243164,
        11.489792823791504,
        19.084699630737305
      ],
      [
        30.029483795166016,
        30.17660903930664,
        24.262208938598633
      ],
      [
        78.4437484741211,
        79.1804428100586,
        49.564693450927734
      ],
      [
        235.00875854492188,
        237.10008239746094,
        153.02981567382812
      ],
      [
        1479.115478515625,
        1488.1566162109375,
        1124.7255859375
      ]
    ],
    "layer_nmse": [
      [
        0.9012316465377808,
        0.9128257036209106,
        0.8623777627944946
      ],
      [
        1.6358941793441772,
        1.742743730545044,
        0.8773683905601501
      ],
      [
        1.4948316812515259,
        1.52433443069458,
        1.0261282920837402
      ],
      [
        1.514439582824707,
        1.5249371528625488,
        1.1338711977005005
      ],
      [
        1.614243984222412,
        1.6153430938720703,
        1.5482895374298096
      ],
      [
        1.5393126010894775,
        1.5327752828598022,
        2.077390670776367
      ],
      [
        2.1999645233154297,
        2.188141345977783,
        3.056568145751953
      ]
    ],
    "insertion_nmse": 0.901231586933136,
    "downstream_mse": 306.6509704589844,
    "epoch": 1,
    "images_seen": 510
  },
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 3.6957764625549316,
      "kl": 3.3756935000419617,
      "insertion_nmse": 0.9228818565607071,
      "downstream_mse": 320.08287048339844,
      "lr": 0.0002974060128544361,
      "gradient_norm": 5.058178901672363,
      "elapsed_seconds": 3.7810000000172295
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 3.7943400144577026,
      "kl": 3.476382076740265,
      "insertion_nmse": 0.8321255296468735,
      "downstream_mse": 317.9578857421875,
      "lr": 0.0002897237368890237,
      "gradient_norm": 5.208731651306152,
      "elapsed_seconds": 7.5
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 3.7510538697242737,
      "kl": 3.4346497654914856,
      "insertion_nmse": 0.9303822815418243,
      "downstream_mse": 316.40411376953125,
      "lr": 0.0002772483976608436,
      "gradient_norm": 5.1818108558654785,
      "elapsed_seconds": 11.094000000011874
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 3.4748103618621826,
      "kl": 3.1574376821517944,
      "insertion_nmse": 0.8540531694889069,
      "downstream_mse": 317.3726501464844,
      "lr": 0.0002604594154601839,
      "gradient_norm": 5.239482402801514,
      "elapsed_seconds": 14.89100000000326
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 4.310755252838135,
      "kl": 3.991400361061096,
      "insertion_nmse": 0.9234306663274765,
      "downstream_mse": 319.3548812866211,
      "lr": 0.0002400019814576463,
      "gradient_norm": 5.418773174285889,
      "elapsed_seconds": 18.64100000000326
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 3.7077375650405884,
      "kl": 3.395490825176239,
      "insertion_nmse": 0.8329333811998367,
      "downstream_mse": 312.2467041015625,
      "lr": 0.00021666226336928708,
      "gradient_norm": 5.024967193603516,
      "elapsed_seconds": 22.26600000000326
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 3.1588839292526245,
      "kl": 2.8441655933856964,
      "insertion_nmse": 0.9305867999792099,
      "downstream_mse": 314.71827697753906,
      "lr": 0.00019133719347217733,
      "gradient_norm": 4.911375999450684,
      "elapsed_seconds": 25.875
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 3.1671265363693237,
      "kl": 2.862201154232025,
      "insertion_nmse": 0.8729004412889481,
      "downstream_mse": 304.92533111572266,
      "lr": 0.000165,
      "gradient_norm": 5.568748474121094,
      "elapsed_seconds": 29.562000000005355
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 3.876264750957489,
      "kl": 3.5683120489120483,
      "insertion_nmse": 0.7744789719581604,
      "downstream_mse": 307.95262908935547,
      "lr": 0.00013866280652782267,
      "gradient_norm": 5.606385707855225,
      "elapsed_seconds": 33.171999999991385
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 3.0171639919281006,
      "kl": 2.703917980194092,
      "insertion_nmse": 0.8442332297563553,
      "downstream_mse": 313.2460174560547,
      "lr": 0.00011333773663071288,
      "gradient_norm": 5.440494060516357,
      "elapsed_seconds": 36.844000000011874
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 3.0316460728645325,
      "kl": 2.7247883081436157,
      "insertion_nmse": 0.8608361333608627,
      "downstream_mse": 306.8576965332031,
      "lr": 8.999801854235373e-05,
      "gradient_norm": 4.84592342376709,
      "elapsed_seconds": 40.594000000011874
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 2.9918046593666077,
      "kl": 2.6782402396202087,
      "insertion_nmse": 0.8742940574884415,
      "downstream_mse": 313.56436920166016,
      "lr": 6.954058453981609e-05,
      "gradient_norm": 5.714993953704834,
      "elapsed_seconds": 44.39100000000326
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 2.8694076538085938,
      "kl": 2.5533387660980225,
      "insertion_nmse": 0.8108172416687012,
      "downstream_mse": 316.06884002685547,
      "lr": 5.275160233915637e-05,
      "gradient_norm": 5.184631824493408,
      "elapsed_seconds": 48.03100000001723
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 3.0249010920524597,
      "kl": 2.7183103561401367,
      "insertion_nmse": 0.8735572695732117,
      "downstream_mse": 306.59071350097656,
      "lr": 4.027626311097629e-05,
      "gradient_norm": 5.357452869415283,
      "elapsed_seconds": 51.64100000000326
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 2.961285889148712,
      "kl": 2.657995343208313,
      "insertion_nmse": 0.8989981710910797,
      "downstream_mse": 303.29051971435547,
      "lr": 3.259398714556389e-05,
      "gradient_norm": 5.364568710327148,
      "elapsed_seconds": 55.15600000001723
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 3.292213757832845,
      "kl": 2.9786242961883547,
      "insertion_nmse": 0.9719147920608521,
      "downstream_mse": 313.5894836425781,
      "lr": 2.9999999999999997e-05,
      "gradient_norm": 6.336579322814941,
      "elapsed_seconds": 58.687000000005355
    }
  ]
}
```
