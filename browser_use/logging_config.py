import logging
import sys


def setup_logging():
	# Prevent duplicate logs by removing any existing handlers
	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)

	# Create formatters
	formatter = logging.Formatter('%(levelname)-8s | %(name)s | %(message)s')

	# Configure console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(formatter)

	# Configure root logger
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.WARNING)  # Default level for third-party logs
	root_logger.addHandler(console_handler)

	# Configure your application's logger
	app_logger = logging.getLogger('browser_use')
	app_logger.setLevel(logging.INFO)
	app_logger.propagate = False  # Prevent propagation to root logger
	app_logger.addHandler(console_handler)

	# Suppress third-party logs
	logging.getLogger('WDM').setLevel(logging.ERROR)
	logging.getLogger('httpx').setLevel(logging.ERROR)
	logging.getLogger('selenium').setLevel(logging.ERROR)
	logging.getLogger('urllib3').setLevel(logging.ERROR)
	logging.getLogger('asyncio').setLevel(logging.ERROR)
