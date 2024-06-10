{
    "cells": [
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "ename": "KeyError",
                    "evalue": "'Parameters'",
                    "output_type": "error",
                    "traceback": [
                        "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
                        "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
                        "Cell \u001b[0;32mIn[3], line 7\u001b[0m\n\u001b[1;32m      4\u001b[0m src_dir \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m../src/\u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[1;32m      5\u001b[0m sys\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mappend(os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mabspath(os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(src_dir)))\n\u001b[0;32m----> 7\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mconfig_params\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Q\n\u001b[1;32m      8\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mcomputations\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m compute_derived_parameters\n",
                        "File \u001b[0;32m~/Documents/Max/Cambridge2324/Lent/Project/src/config_params.py:7\u001b[0m\n\u001b[1;32m      4\u001b[0m config \u001b[38;5;241m=\u001b[39m load_config(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mconfig.ini\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m      6\u001b[0m \u001b[38;5;66;03m# --- Parameter configuration\u001b[39;00m\n\u001b[0;32m----> 7\u001b[0m INCLUDE_PLANET_B \u001b[38;5;241m=\u001b[39m \u001b[43mconfig\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mParameters\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[38;5;241m.\u001b[39mgetboolean(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mINCLUDE_PLANET_B\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m      9\u001b[0m params_general \u001b[38;5;241m=\u001b[39m [\n\u001b[1;32m     10\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mA_RV\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mP_rot\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mt_decay\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgamma\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m     11\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124msigma_RV_pre\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124msigma_RV_post\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124msigma_RV_harps\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m     12\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mv0_pre\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124moff_post\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124moff_harps\u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[1;32m     13\u001b[0m ]\n\u001b[1;32m     15\u001b[0m params_general_latex \u001b[38;5;241m=\u001b[39m [\n\u001b[1;32m     16\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mA_\u001b[39m\u001b[38;5;132;01m{RV}\u001b[39;00m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mP_\u001b[39m\u001b[38;5;132;01m{rot}\u001b[39;00m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mt_\u001b[39m\u001b[38;5;132;01m{decay}\u001b[39;00m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124mgamma\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m     17\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124msigma_\u001b[39m\u001b[38;5;124m{\u001b[39m\u001b[38;5;124mRV, pre}\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124msigma_\u001b[39m\u001b[38;5;124m{\u001b[39m\u001b[38;5;124mRV, post}\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124msigma_\u001b[39m\u001b[38;5;124m{\u001b[39m\u001b[38;5;124mRV, HARPS}\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m     18\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mv_\u001b[39m\u001b[38;5;124m{\u001b[39m\u001b[38;5;124m0, pre}\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124moff_\u001b[39m\u001b[38;5;132;01m{post}\u001b[39;00m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124moff_\u001b[39m\u001b[38;5;132;01m{HARPS}\u001b[39;00m\u001b[38;5;124m'\u001b[39m\n\u001b[1;32m     19\u001b[0m ]\n",
                        "File \u001b[0;32m~/miniconda3/envs/l9859-env/lib/python3.8/configparser.py:960\u001b[0m, in \u001b[0;36mRawConfigParser.__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m    958\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__getitem__\u001b[39m(\u001b[38;5;28mself\u001b[39m, key):\n\u001b[1;32m    959\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m key \u001b[38;5;241m!=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdefault_section \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhas_section(key):\n\u001b[0;32m--> 960\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key)\n\u001b[1;32m    961\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_proxies[key]\n",
                        "\u001b[0;31mKeyError\u001b[0m: 'Parameters'",
                    ],
                }
            ],
            "source": [
                "import os\n",
                "import sys\n",
                "\n",
                "src_dir = '../src/'\n",
                "sys.path.append(os.path.abspath(os.path.join(src_dir)))\n",
                "\n",
                "from config_params import Q\n",
                "from computations import compute_derived_parameters",
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": []},
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "l9859-env",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.18",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}
