import logging
import sys


def setup_logging():
	# Clear existing handlers
	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)

	class ModuleNameFormatter(logging.Formatter):
		def format(self, record):
			# Get the last part of the module name (after the last dot)
			if record.name.startswith('browser_use.'):
				record.name = record.name.split('.')[-2]  # Get the second-to-last part
			return super().format(record)

	# Create formatter with custom class
	formatter = ModuleNameFormatter('%(levelname)-8s [%(name)s] %(message)s')

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
	app_logger.propagate = False  # Prevent duplicate logs
	app_logger.addHandler(console_handler)

	# Suppress third-party logs
	for logger_name in ['WDM', 'httpx', 'selenium', 'urllib3', 'asyncio']:
		logging.getLogger(logger_name).setLevel(logging.ERROR)
