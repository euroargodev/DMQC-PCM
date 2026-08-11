#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import argopy
from pyowc import calibration, configuration, plot
from argodmqc_pcm.PCM_utils_forDMQC.BIC_calculation import plot_BIC
from argodmqc_pcm.PCM_utils_forDMQC.classification import applyBIC, applyPCM, loadReferenceData, setupLogger
from argodmqc_pcm.PCM_utils_forDMQC.config_context import config_context
from argopy import DataFetcher as ArgoDataFetcher


class SO_DMQC:
    def __init__(self, config_file):
        with open(config_file) as file:
            config = json.loads(file.read())

        self.PCM_CONFIG = config["PCM"]
        self.OWC_CONFIG = config["OWC"]

        self.LOGS_DIR = config["OUTPUT DIRECTORIES"]["LOGS_DIR"]

        self.FLOAT_SOURCE_RAW = self.OWC_CONFIG["FLOAT_SOURCE_DIRECTORY"]
        self.FLOAT_SOURCE_ADJUSTED = self.FLOAT_SOURCE_RAW.replace("default", "adjusted")
        self.CONFIG_DIR = self.OWC_CONFIG["CONFIG_DIRECTORY"]
        self.ELEVATION_FILE = self.OWC_CONFIG["ELEVATION_FILE"]

        self.ROLE = config["ROLE"]

        self._full_config = {
            "PCM": self.PCM_CONFIG,
            "OWC": self.OWC_CONFIG,
            "ROLE": self.ROLE,
            "ELEVATION_FILE": self.ELEVATION_FILE,
        }

    def create_wong_matrix(self, float_wmo, logger):
        data_src = self.PCM_CONFIG.get("SRC", "gdac").lower()
        if data_src == "localftp":
            # local copy of GDAC
            with argopy.set_options(src="gdac", gdac=self.PCM_CONFIG["GDAC_MIRROR"], mode="expert"):
                ds = ArgoDataFetcher().float(float_wmo).load().data

        elif data_src == "gdac":
            with argopy.set_options(src="gdac", gdac=self.PCM_CONFIG["GDAC"], mode="expert"):
                ds = ArgoDataFetcher().float(float_wmo).load().data

        else:
            # erddap, argovis etc
            with argopy.set_options(src=data_src, mode="expert"):
                ds = ArgoDataFetcher().float(float_wmo).load().data

        if self.ROLE == "auditor":  # Attempt to create source using Adjusted data
            ds.argo.create_float_source(self.FLOAT_SOURCE_ADJUSTED, force="adjusted")
            ds.argo.create_float_source(self.FLOAT_SOURCE_RAW)
            logger.info("Success: Adjusted and Raw source created.")
        elif self.ROLE == "operational":
            ds.argo.create_float_source(self.FLOAT_SOURCE_RAW)
            logger.info("Success: Raw data source created")

    def run_pcm(self, logger, wong_matrix_path, float_wmo):
        logger.info("applying PCM")
        logger.info(">>> loading reference data")

        # Starting the PCM analysis
        logger.info("Computing reference dataset...")
        ds = loadReferenceData(
            float_mat_path=wong_matrix_path,
            ow_config=self.OWC_CONFIG,
        )
        ds = ds.compute()  # ensure no lazy graph
        logger.info(">>> applying BIC")

        # apply BIC function to determine most suitable number of classes
        BIC, number_classes = applyBIC(
            ds=ds,
            Nrun=int(self.PCM_CONFIG["NUMBER_RUNS"]),
            NK=int(self.PCM_CONFIG["NK"]),
            corr_dist=int(self.PCM_CONFIG["CORR_DISTANCE"]),
            max_depth=int(self.PCM_CONFIG["MAX_DEPTH"]),
            logger_name=logger.name,
        )

        logger.info(f">>> classes: {number_classes}")

        # generate the BIC plot
        plot_BIC(
            BIC=BIC,
            NK=int(self.PCM_CONFIG["NK"]),
            float_WMO=float_wmo,
            plots_dir=self.PCM_CONFIG["PLOTS_DIR"],
        )
        logger.info(">>> BIC successful")

        logger.info(">>> classifying")
        pcm_file_path = (
            self.PCM_CONFIG["CLASSES_DIR"] + f"PCM_classes_{float_wmo}_K{number_classes}.txt"
        )
        logger.info("pcm file will be saved to %s", pcm_file_path)
        # run PCM function to calculate the classes and save OWC text file
        applyPCM(
            ds=ds,
            float_WMO=float_wmo,
            float_mat_path=wong_matrix_path,
            pcm_file_path=pcm_file_path,
            number_classes=number_classes,
            corr_dist=int(self.PCM_CONFIG["CORR_DISTANCE"]),
            max_depth=int(self.PCM_CONFIG["MAX_DEPTH"]),
            plots_dir=self.PCM_CONFIG["PLOTS_DIR"],
            models_dir=self.PCM_CONFIG["MODELS_DIR"],
        )

        logger.info(">>> applying PCM successful")

    def run(self, float_list: list[str]):
        # process each float in the list of WMO numbers in turn
        with config_context(self._full_config):
            for float_WMO in float_list:
                ##### Setup logger #####
                log_file_path = f"{self.LOGS_DIR}{float_WMO}_runtime_log.txt"
                logger_name = f"{float_WMO}_runtime_logger"
                logger = setupLogger(logger_name=logger_name, log_file=log_file_path, level=logging.INFO)
                info_message = "starting processing"
                logger.info(f"{info_message} WMO number: {float_WMO}")

                ##### Generate Wong matrix for raw data #####
                wong_matrix_path = os.path.join(self.FLOAT_SOURCE_RAW, f"{float_WMO}.mat")
                if not os.path.exists(wong_matrix_path):
                    self.create_wong_matrix(float_WMO, logger)
                    logger.info("Wong matrix created")
                else:
                    logger.info("Wong matrix already exists")

                ##### Apply PCM #####
                pcm_file_root = f"PCM_classes_{float_WMO}"
                PCM_file_name = []
                classes_dir = Path(self.PCM_CONFIG["CLASSES_DIR"])
                if classes_dir.is_dir():
                    PCM_file_name = [
                        file_name
                        for file_name in Path.iterdir(classes_dir)
                        if file_name.name[: len(pcm_file_root)] == pcm_file_root
                    ]
                else:
                    logger.info(f"CLASSES_DIR does not exist: {classes_dir} - creating")
                    os.makedirs(classes_dir, exist_ok=True)
                # if the PCM output file already exists skip this float
                # NB that the number of classes in the output file name can vary
                if PCM_file_name:
                    logger.info("PCM has already been run - skipping")
                else:
                    self.run_pcm(logger, wong_matrix_path, float_WMO)

                ##### Apply OWC #####
                logger.info("applying OWC")
                logger.info(">>> updating salinity mapper")
                calibration.update_salinity_mapping("", self.OWC_CONFIG, str(float_WMO), self.PCM_CONFIG["CLASSES_DIR"])
                logger.info(">>> setting cal series")
                configuration.set_calseries("", str(float_WMO), self.OWC_CONFIG)
                logger.info(">>> calculating piecewise fit")
                fit_type = calibration.calc_piecewisefit("", str(float_WMO), self.OWC_CONFIG)
                logger.info(f">>> fit type {fit_type}")
                logger.info(">>> generating plots")
                plot.dashboard("", str(float_WMO), self.OWC_CONFIG, headless=True)
                logger.info(">>> OWC successful")

                logging.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SO-DMQC assessment.")
    parser.add_argument("floats", metavar="WMO", type=int, nargs="*", help="One or more WMO float numbers")

    args = parser.parse_args()
    float_list = args.floats
    # turn off console warnings
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")

    # Make an instance of the class and implement the run function
    obj = SO_DMQC("pcm_owc_config.json")
    obj.run(float_list)
