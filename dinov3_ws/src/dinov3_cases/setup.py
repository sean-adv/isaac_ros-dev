from setuptools import setup
setup(name="dinov3_cases", version="0.1.0", packages=["dinov3_cases"], data_files=[("share/ament_index/resource_index/packages", ["resource/dinov3_cases"]), ("share/dinov3_cases", ["package.xml"])], entry_points={"console_scripts": ["case_heatmap = dinov3_cases.node:main"]})
