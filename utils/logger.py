#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

LOGS_PATH = Path("logs")
LOGS_PATH.mkdir(parents=True, exist_ok=True)


def get_logger(name: Optional[str] = None, log_to_file: bool = True, level: int = logging.INFO) -> logging.Logger:
    
    if name is None:
        script_name = sys.argv[0]
        name = (
            script_name.split("/")[-1].split(".py")[0]
            if "/" in script_name
            else script_name.split("\\")[-1].split(".py")[0]
        )
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        # "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_to_file:
        log_file = LOGS_PATH / f"{name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info("=" * 90)
        logger.info(f"Execution of '{name}' - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 90)
    
    return logger


def get_script_logger(log_to_file: bool = True, level: int = logging.INFO) -> logging.Logger:
    return get_logger(name=None, log_to_file=log_to_file, level=level)


logger = get_script_logger()