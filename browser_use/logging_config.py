import logging
import os
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
				module_parts = record.name.split('.')
				if len(module_parts) >= 3:  # browser_use.agent.service
					record.name = module_parts[-2]  # Get 'agent'
				else:
					record.name = module_parts[-1]  # Fallback to last part
			return super().format(record)

	# Create formatter with custom class
	formatter = ModuleNameFormatter('%(levelname)-8s [%(name)s] %(message)s')

	# Configure console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(formatter)

	# Configure root logger
	root_logger = logging.getLogger()
	root_logger.setLevel(os.getenv('PYTEST_LOG_LEVEL', logging.INFO))
	root_logger.addHandler(console_handler)

	# Configure your application's logger
	app_logger = logging.getLogger('browser_use')
	app_logger.setLevel(os.getenv('PYTEST_LOG_LEVEL', logging.INFO))
	app_logger.propagate = False
	app_logger.addHandler(console_handler)

	# Suppress third-party logs but respect pytest settings
	third_party_level = logging.ERROR if not os.getenv('PYTEST_DEBUG') else logging.INFO
	for logger_name in ['WDM', 'httpx', 'selenium', 'urllib3', 'asyncio']:
		logging.getLogger(logger_name).setLevel(third_party_level)
