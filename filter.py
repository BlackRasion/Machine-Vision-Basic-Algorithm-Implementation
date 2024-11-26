from ipywidgets import widgets
# 滤波器参数控件，默认隐藏
filter_params = widgets.HBox([
    widgets.IntSlider(min=1, max=15, step=2, value=5, description='Kernel Size:', layout={'visibility': 'hidden'}),
    widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.5, description='Sigma X:', layout={'visibility': 'hidden'}),
    widgets.IntSlider(min=1, max=15, step=2, value=9, description='Diameter:', layout={'visibility': 'hidden'}),
    widgets.IntSlider(min=1, max=100, step=1, value=75, description='Sigma Color:', layout={'visibility': 'hidden'}),
    widgets.IntSlider(min=1, max=100, step=1, value=75, description='Sigma Space:', layout={'visibility': 'hidden'})
])