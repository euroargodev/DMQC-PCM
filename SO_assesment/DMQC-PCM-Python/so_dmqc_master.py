#!/usr/bin/env python
import argparse
import configparser
import logging
import os
import sys
from pathlib import Path

import argopy
import xarray as xr
from pyowc import calibration, configuration, plot
from argodmqc_pcm.PCM_utils_forDMQC.BIC_calculation import plot_BIC
from argodmqc_pcm.PCM_utils_forDMQC.classification import applyBIC, applyPCM, loadReferenceData, setupLogger
from argodmqc_pcm.PCM_utils_forDMQC.config_context import config_context
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import DataNotFound, NetCDF4FileNotFoundError


class SO_DMQC:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        print(config_file)
        config.read(config_file)
        base_dir = Path(__file__).resolve().parent

        self.PCM_CONFIG = config["PCM"]
        self.OWC_CONFIG = config["OWC"]

        self.LOGS_DIR = config["OUTPUT DIRECTORIES"]["LOGS_DIR"]
        self.DAC_COMP_DIR = config["OUTPUT DIRECTORIES"]["DAC_COMP_DIR"]

        self.FLOAT_SOURCE_RAW = self.OWC_CONFIG["FLOAT_SOURCE_DIRECTORY"]
        self.FLOAT_SOURCE_ADJUSTED = self.FLOAT_SOURCE_RAW.replace("default", "adjusted")
        self.CACHE_DIR = base_dir / config["OUTPUT DIRECTORIES"]["CACHE_DIR"]
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.CONFIG_DIR = self.OWC_CONFIG["CONFIG_DIRECTORY"]
        self.ELEVATION_FILE = self.OWC_CONFIG["ELEVATION_FILE"]

        self.ROLE = self.OWC_CONFIG["ROLE"]

        self._full_config = {
            "PCM": self.PCM_CONFIG,
            "OWC": self.OWC_CONFIG,
            "ROLE": self.ROLE,
            "ELEVATION_FILE": self.ELEVATION_FILE,
            "OUTPUT_DIRECTORIES": {
                "CACHE_DIR": str(self.CACHE_DIR),
                "LOGS_DIR": self.LOGS_DIR,
                "DAC_COMP_DIR": self.DAC_COMP_DIR,
            },
        }

    def _should_regenerate_cache(self, cache_file: Path, wong_matrix_path: Path, logger) -> bool:
        """Regenerate cache if it doesn't exist or source data is newer."""
        runtime_logger = logger or logging.getLogger(__name__)
        if not cache_file.exists():
            runtime_logger.info("Cache file does not exist, will generate.")
            return True

        cache_mtime = cache_file.stat().st_mtime
        source_mtime = wong_matrix_path.stat().st_mtime

        if source_mtime > cache_mtime:
            runtime_logger.info(
                "Source data newer than cache (source: %s, cache: %s), regenerating.",
                datetime.fromtimestamp(source_mtime),
                datetime.fromtimestamp(cache_mtime),
            )
            return True

        cache_size = cache_file.stat().st_size
        if cache_size < 1024:  # suspiciously small — likely corrupt/empty
            runtime_logger.warning("Cache file suspiciously small (%d bytes), regenerating.", cache_size)
            return True

        runtime_logger.info("Cache is up to date (%d bytes), loading from cache.", cache_size)
        return False

    def run(self):
        # process each float in the list of WMO numbers in turn
        with config_context(self._full_config):
            for float_WMO in float_list:
                run_PCM_flag = True

                log_file_path = f"{self.LOGS_DIR}{float_WMO}_runtime_log.txt"
                logger_name = f"{float_WMO}_runtime_logger"
                runtime_logger = setupLogger(logger_name=logger_name, log_file=log_file_path, level=logging.INFO)
                info_message = "starting processing"
                runtime_logger.info(f"{info_message} WMO number: {float_WMO}")
                runtime_logger.info(info_message)

                # Generating Wong matrix for raw data
                wong_matrix_path = os.path.join(self.FLOAT_SOURCE_RAW, f"{float_WMO}.mat")

                if not os.path.exists(wong_matrix_path):
                    try:
                        data_src = self.PCM_CONFIG.get("SRC", "gdac").lower()
                        if data_src == "localftp":
                            # local copy of GDAC
                            with argopy.set_options(src="gdac", gdac=self.PCM_CONFIG["GDAC_MIRROR"], mode="expert"):
                                ds = ArgoDataFetcher().float(float_WMO).load().data

                        elif data_src == "gdac":
                            with argopy.set_options(src="gdac", gdac=self.PCM_CONFIG["GDAC"], mode="expert"):
                                ds = ArgoDataFetcher().float(float_WMO).load().data

                        else:
                            # erddap, argovis etc
                            with argopy.set_options(src=data_src, mode="expert"):
                                ds = ArgoDataFetcher().float(float_WMO).load().data

                        if self.ROLE == "auditor":  # Attempt to create source using Adjusted data
                            ds.argo.create_float_source(self.FLOAT_SOURCE_ADJUSTED, force="adjusted")
                            ds.argo.create_float_source(self.FLOAT_SOURCE_RAW)
                            print("Success: Adjusted and Raw source created.")
                        elif self.ROLE == "operational":
                            ds.argo.create_float_source(self.FLOAT_SOURCE_RAW)
                            print("Success: Raw data source created")

                    except DataNotFound:
                        error_message = "XXX DataNotFound error: due to QC flags (check netCDF file) - skipping"

                        runtime_logger.info(error_message)
                        logging.shutdown()
                        continue
                    except ValueError as e:
                        error_message = f"XXX ValueError: check whether WMO is valid :{e!s}"
                        print(error_message + " - skipping")
                        runtime_logger.info(error_message)
                        logging.shutdown()
                        continue
                    except NetCDF4FileNotFoundError:
                        error_message = "XXX NetCDF4FileNotFoundError: check whether WMO is valid"
                        print(error_message + " - skipping")
                        runtime_logger.info(error_message)
                        logging.shutdown()
                        continue

                    info_message = "Wong matrix created"
                else:
                    info_message = "Wong matrix already exists"

                print(info_message)
                runtime_logger.info(info_message)

                # if the PCM output file already exists skip this float
                # NB that the number of classes in the output file name can vary
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
                    runtime_logger.warning(f"CLASSES_DIR does not exist: {classes_dir}")
                    os.makedirs(classes_dir, exist_ok=True)

                if PCM_file_name:
                    run_PCM_flag = False
                    if int(self.OWC_CONFIG["USE_PCM"]):
                        error_message = "PCM has already been run"
                        print(error_message + " - skipping ")
                        runtime_logger.info(error_message)

                if run_PCM_flag and int(self.OWC_CONFIG["USE_PCM"]):
                    info_message = "applying PCM"
                    print(info_message)
                    runtime_logger.info(info_message)
                    info_message = ">>> loading reference data"
                    print(info_message)
                    runtime_logger.info(info_message)

                    # Starting the PCM analysis
                    cache_file = Path(f"{self.CACHE_DIR}/cache_{float_WMO}.nc")
                    if self._should_regenerate_cache(cache_file, Path(wong_matrix_path), logger=runtime_logger):
                        runtime_logger.info("Computing reference dataset...")
                        ds = loadReferenceData(
                            float_mat_path=wong_matrix_path,
                            ow_config=self.OWC_CONFIG,
                        )
                        ds = ds.compute()  # ensure no lazy graph
                        ds.to_netcdf(cache_file)
                        runtime_logger.info("Written file exists? %s", cache_file.exists())
                    else:
                        runtime_logger.info("Loading cached dataset...")
                        ds = xr.open_dataset(cache_file)

                    info_message = ">>> applying BIC"
                    print(info_message)
                    runtime_logger.info(info_message)

                    # apply BIC function to determine most suitable number of classes
                    BIC, number_classes = applyBIC(
                        ds=ds,
                        Nrun=int(self.PCM_CONFIG["NUMBER_RUNS"]),
                        NK=int(self.PCM_CONFIG["NK"]),
                        corr_dist=int(self.PCM_CONFIG["CORR_DISTANCE"]),
                        max_depth=int(self.PCM_CONFIG["MAX_DEPTH"]),
                        logger_name=logger_name,
                    )

                    runtime_logger.info(f">>> classes: {number_classes}")

                    # generate the BIC plot
                    plot_BIC(
                        BIC=BIC,
                        NK=int(self.PCM_CONFIG["NK"]),
                        float_WMO=float_WMO,
                        plots_dir=self.PCM_CONFIG["PLOTS_DIR"],
                    )
                    runtime_logger.info(">>> successful")

                    try:
                        info_message = ">>> classifying"
                        print(info_message)
                        runtime_logger.info(info_message)
                        pcm_file_path = (
                            self.PCM_CONFIG["CLASSES_DIR"] + f"PCM_classes_{float_WMO}_K{number_classes}.txt"
                        )
                        print("pcm file will be saved to %s", pcm_file_path)
                        # run PCM function to calculate the classes and save OWC text file
                        applyPCM(
                            ds=ds,
                            float_WMO=float_WMO,
                            float_mat_path=wong_matrix_path,
                            pcm_file_path=pcm_file_path,
                            number_classes=number_classes,
                            corr_dist=int(self.PCM_CONFIG["CORR_DISTANCE"]),
                            max_depth=int(self.PCM_CONFIG["MAX_DEPTH"]),
                            plots_dir=self.PCM_CONFIG["PLOTS_DIR"],
                            models_dir=self.PCM_CONFIG["MODELS_DIR"],
                        )

                        runtime_logger.info(">>> successful")

                    except IndexError:
                        error_message = "XXX IndexError: too few profiles in data_fetcher.py to index array"
                        print(error_message)
                        runtime_logger.info(error_message)
                    except MemoryError:
                        error_message = "XXX ArrayMemoryError: Unable to allocate sufficient memory"
                        print(error_message + ' for numpy array in "pyxpcm/xarray.py"')
                        runtime_logger.info(error_message)

                info_message = "applying OWC"
                print(info_message)
                runtime_logger.info(info_message)
                try:
                    info_message = ">>> updating salinity mapper"
                    print(info_message)
                    runtime_logger.info(info_message)
                    calibration.update_salinity_mapping(str(float_WMO), self.OWC_CONFIG, self.PCM_CONFIG["CLASSES_DIR"])
                    runtime_logger.info(">>> successful")

                except FileNotFoundError:
                    error_message = "XXX file not found - check float source"
                    print(error_message)
                    runtime_logger.info(error_message)
                    continue
                except RuntimeError:
                    error_message = 'XXX "NO DATA FOUND" - most likely calibration.get_region_data'
                    print(error_message)
                    runtime_logger.info(error_message)
                    continue

                try:
                    info_message = ">>> setting cal series"
                    print(info_message)
                    runtime_logger.info(info_message)
                    configuration.set_calseries(str(float_WMO), self.OWC_CONFIG)
                    runtime_logger.info(">>> successful")

                except FileNotFoundError:
                    error_message = "XXX file not found - check float mapped"
                    print(error_message)
                    runtime_logger.info(error_message)
                    continue

                try:
                    info_message = ">>> calculating piecewise fit"
                    print(info_message)
                    runtime_logger.info(info_message)
                    fit_type = calibration.calc_piecewisefit(str(float_WMO), self.OWC_CONFIG)
                    runtime_logger.info(f">>> fit type {fit_type}")
                    runtime_logger.info(">>> successful")

                except AttributeError:
                    error_message = "XXX most likely no good data was found in core.stats.fit_cond"
                    print(error_message)
                    runtime_logger.info(error_message)
                    continue
                except ValueError:
                    error_message = "XXX issue fitting with breaks in pyowc.core.stats.fit_cond"
                    print(error_message)
                    runtime_logger.info(error_message)
                    continue

                try:
                    info_message = ">>> generating plots"
                    print(info_message)
                    runtime_logger.info(info_message)
                    plot.dashboard(str(float_WMO), self.OWC_CONFIG)
                    runtime_logger.info(">>> successful")

                except FileNotFoundError:
                    error_message = "XXX file not found - either float source, mapped or calibrated"
                    print(error_message)
                    runtime_logger.info(error_message)

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
    obj = SO_DMQC("pcm_owc_config.ini")
    obj.run()
