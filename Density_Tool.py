#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import (
							FigureCanvasQTAgg as FigureCanvas,
							NavigationToolbar2QT as NavigationToolbar
							)
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors, ticker, colormaps
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import Delaunay, Voronoi, KDTree
from scipy.interpolate import make_interp_spline
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QIntValidator, QMouseEvent
from PyQt5.QtWidgets import (
							QApplication, QLabel, QWidget,
							QPushButton, QHBoxLayout, QVBoxLayout,
							QComboBox, QCheckBox, QSlider, QProgressBar,
							QFormLayout, QLineEdit, QTabWidget,
							QSizePolicy, QFileDialog, QMessageBox,
							QFrame
							)
from pathlib import Path

################################################################################
# helper functions for GUI elements #
#####################################

def display_error (error_text = 'Something went wrong!'):
	msg = QMessageBox()
	msg.setIcon(QMessageBox.Critical)
	msg.setText("Error")
	msg.setInformativeText(error_text)
	msg.setWindowTitle("Error")
	msg.exec_()

def setup_textbox (function, layout, label_text,
				   initial_value = 0):
	textbox = QLineEdit()
	need_inner = not isinstance(layout, QHBoxLayout)
	if need_inner:
		inner_layout = QHBoxLayout()
	label = QLabel(label_text)
	label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
	if need_inner:
		inner_layout.addWidget(label)
	else:
		layout.addWidget(label)
	textbox.setMaxLength(4)
	textbox.setFixedWidth(50)
	textbox.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
	textbox.setValidator(QIntValidator())
	textbox.setText(str(initial_value))
	textbox.editingFinished.connect(function)
	if need_inner:
		inner_layout.addWidget(textbox)
		layout.addLayout(inner_layout)
	else:
		layout.addWidget(textbox)
	return textbox

def get_textbox (textbox,
				 minimum_value = None,
				 maximum_value = None,
				 is_int = False):
	if is_int:
		value = int(np.floor(float(textbox.text())))
	else:
		value = float(textbox.text())
	if maximum_value is not None:
		if value > maximum_value:
			value = maximum_value
	if minimum_value is not None:
		if value < minimum_value:
			value = minimum_value
	textbox.setText(str(value))
	return value

def setup_button (function, layout, label_text, toggle = False):
	button = QPushButton()
	if toggle:
		button.setCheckable(True)
	button.setText(label_text)
	button.clicked.connect(function)
	layout.addWidget(button)
	return button

def setup_checkbox (function, layout, label_text,
					is_checked = False):
		checkbox = QCheckBox()
		checkbox.setText(label_text)
		checkbox.setChecked(is_checked)
		checkbox.stateChanged.connect(function)
		layout.addWidget(checkbox)
		return checkbox

def setup_tab (tabs, tab_layout, label):
	tab = QWidget()
	tab.layout = QVBoxLayout()
	tab.setLayout(tab.layout)
	tab.layout.addLayout(tab_layout)
	tabs.addTab(tab, label)

def horizontal_separator (layout, palette):
	separator = QFrame()
	separator.setFrameShape(QFrame.HLine)
	#separator.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Expanding)
	separator.setLineWidth(1)
	palette.setColor(QPalette.WindowText, QColor('lightgrey'))
	separator.setPalette(palette)
	layout.addWidget(separator)

def setup_progress_bar (layout):
	progress_bar = QProgressBar()
	clear_progress_bar(progress_bar)
	layout.addWidget(progress_bar)
	return progress_bar

def clear_progress_bar (progress_bar):
	progress_bar.setMinimum(0)
	progress_bar.setFormat('')
	progress_bar.setMaximum(1)
	progress_bar.setValue(0)

def update_progress_bar (progress_bar, value = None,
						 minimum_value = None,
						 maximum_value = None,
						 text = None):
	if minimum_value is not None:
		progress_bar.setMinimum(minimum_value)
	if maximum_value is not None:
		progress_bar.setMaximum(maximum_value)
	if value is not None:
		progress_bar.setValue(value)
	if text is not None:
		progress_bar.setFormat(text)

def setup_slider (layout, function, maximum_value = 1,
				  direction = Qt.Horizontal):
		slider = QSlider(direction)
		slider.setMinimum(0)
		slider.setMaximum(maximum_value)
		slider.setSingleStep(1)
		slider.setValue(0)
		slider.valueChanged.connect(function)
		return slider

def update_slider (slider, value = None,
				   maximum_value = None):
	if value is not None:
		slider.setValue(value)
	if maximum_value is not None:
		slider.setMaximum(maximum_value)

def setup_combobox (function, layout, label_text):
	combobox = QComboBox()
	need_inner = not isinstance(layout, QHBoxLayout)
	if need_inner:
		inner_layout = QHBoxLayout()
	label = QLabel(label_text)
	label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
	if need_inner:
		inner_layout.addWidget(label)
	else:
		layout.addWidget(label)
	combobox.currentIndexChanged.connect(function)
	if need_inner:
		inner_layout.addWidget(combobox)
		layout.addLayout(inner_layout)
	else:
		layout.addWidget(combobox)
	return combobox

def setup_labelbox (label_text, initial_text):
	text_box = QFrame()
	layout = QHBoxLayout()
	text_box.setFrameShape(QFrame.StyledPanel)
#	self.instruction_box.setSizePolicy(QSizePolicy.Expanding)
	label = QLabel(label_text)
	label.setAlignment(Qt.AlignLeft)
	text = QLabel(initial_text)
	text.setAlignment(Qt.AlignLeft)
#	self.instruction_text.setWordWrap(True)
	layout.addWidget(label)
	layout.addWidget(text)
	layout.addStretch()
	text_box.setLayout(layout)
	return text_box, text

def clear_layout (layout):
	for i in reversed(range(layout.count())): 
		widgetToRemove = layout.takeAt(i).widget()
		layout.removeWidget(widgetToRemove)
		widgetToRemove.deleteLater()

################################################################################
# find bright points in image array #
#####################################

def find_centres (frame, neighbourhood_size = 16,
				  threshold_difference = 1, gauss_deviation = 2, channel = 0):
	x_size, y_size = frame.shape[0:2]
	if len(frame.shape) == 3:
		frame = frame[:,:,channel]
	frame = np.random.uniform(low = 0.0, high = 1e-5, size = frame.shape) + \
			np.astype(frame, float)
	frame = ndi.gaussian_filter(frame, gauss_deviation)
	frame_max = ndi.maximum_filter(frame, neighbourhood_size)
	maxima = (frame == frame_max)
	frame_min = ndi.minimum_filter(frame, neighbourhood_size)
	differences = ((frame_max - frame_min) > threshold_difference)
	maxima[differences == 0] = 0
	maximum = np.amax(frame)
	minimum = np.amin(frame)
	outside_filter = (frame_max > (maximum-minimum)*0.1 + minimum)
	maxima[outside_filter == 0] = 0
	labeled, num_objects = ndi.label(maxima)
	slices = ndi.find_objects(labeled)
	centres = np.zeros((len(slices),2), dtype = int)
	good_centres = 0
	for (dy,dx) in slices:
		centres[good_centres,0] = int((dx.start + dx.stop - 1)/2)
		centres[good_centres,1] = int((dy.start + dy.stop - 1)/2)
		if centres[good_centres,0] < neighbourhood_size/2 or \
		   centres[good_centres,0] > y_size - neighbourhood_size/2 or \
		   centres[good_centres,1] < neighbourhood_size/2 or \
		   centres[good_centres,1] > x_size - neighbourhood_size/2:
			good_centres -= 1
		good_centres += 1
	centres = centres[:good_centres]
#	to_remove = np.zeros(centres.shape[0])
#	for i, centre_i in enumerate(centres):
#		if to_remove[i]:
#			continue
#		for j, centre_j in enumerate(centres):
#			if i == j:
#				continue
#			if to_remove[j]:
#				continue
#			if np.linalg.norm(centre_i-centre_j) < neighbourhood_size/2:
#				centre_i = (centre_i + centre_j)/2
#				to_remove[j] = 1
#	centres = centres[to_remove == 0]
	return centres

################################################################################
# read multichannel tiff #
##########################

def read_tiff(path):
	img = Image.open(path)
	images = []
	for i in range(img.n_frames):
		img.seek(i)
		images.append(np.array(img))
	return np.array(images)

################################################################################
# returns area of a polygon #
#############################

def PolyArea(x,y):
#	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
	return 0.5*np.abs(np.dot(x,np.roll(y,1)-np.roll(y,-1)))

################################################################################
# matplotlib canvas widget #
############################

class MPLCanvas(FigureCanvas):
	def __init__ (self, parent=None, width=8, height=8, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.add_subplot(111)
		self.ax.set_facecolor('black')
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		self.image = None
		self.points = None
		self.voronoi = None
		self.areas = None
		self.path = None
		self.image_plot = None
		self.points_plot = None
		self.lines_plot = None
		self.areas_plot = None
		self.path_plot = None
		self.show_image = True
		self.show_points = True
		self.show_voronoi = True
		self.show_path = True
		self.image_changed = False
		self.points_changed = False
		self.voronoi_changed = False
		self.path_changed = False
	
	def plot_image (self):
		if self.image is None:
			return False
		if not self.image_changed:
			return False
		self.image_changed = False
		self.remove_plot_element(self.image_plot)
		self.image_plot = None
		if self.show_image:
			self.image_plot = self.ax.imshow(self.image,
											 cmap='Grays_r',
											 zorder=5)
	
	def plot_path (self):
		if self.path is None:
			return False
		if not self.path_changed:
			return False
		self.path_changed = False
		self.remove_plot_element(self.path_plot)
		self.path_plot = None
		if self.show_path:
			self.path_plot = []
			self.path_plot.append(self.ax.plot(
											self.path[:,0],
											self.path[:,1],
											linestyle = '-',
											marker = '',
											color = 'tab:orange',
											zorder=9
												))
			self.path_plot.append(self.ax.plot(
											self.path[-1,0],
											self.path[-1,1],
											linestyle = '',
											marker = 'o',
											color = 'tab:orange',
											zorder=9
												))
	
	def plot_points (self):
		if self.points is None:
			return False
		if not self.points_changed:
			return False
		self.points_changed = False
		self.remove_plot_element(self.points_plot)
		self.points_plot = None
		if self.show_points:
#		if self.voronoi is None or self.areas is None:
			if len(self.points.shape) == 2:
				self.points_plot = []
				self.points_plot.append(self.ax.plot(
												self.points[:,0],
												self.points[:,1],
												linestyle = '',
												marker = 'o',
												markersize = 5000. / \
													self.image.shape[0],
												color = 'white',
												zorder=6)
													)
				self.points_plot.append(self.ax.plot(
												self.points[:,0],
												self.points[:,1],
												linestyle = '',
												marker = '.',
												markersize = 4000. / \
													self.image.shape[0],
												color = 'magenta',
												zorder=7)
													)
	
	def plot_voronoi (self):
		if self.points is None:
			return False
		if self.voronoi is None:
			return False
		if self.areas is None:
			return False
		if not self.voronoi_changed:
			return False
		self.voronoi_changed = False
#		self.remove_plot_element(self.lines_plot)
#		self.lines_plot = None
		self.remove_plot_element(self.areas_plot)
		self.areas_plot = None
		if not self.show_voronoi:
			return False
		area_min = np.amin(self.areas[self.areas>0])
		area_max = np.amax(self.areas[self.areas>0])
		area_mid = np.median(self.areas[self.areas>0])
		cmap = matplotlib.colormaps.get_cmap('viridis_r')
#		cmap = matplotlib.colormaps.get_cmap('afmhot_r')
		norm = matplotlib.colors.TwoSlopeNorm(vcenter = area_mid,
											vmin = area_min,
											vmax = area_max)
		points = self.voronoi.vertices
		self.areas_plot = []
		for index, centre in enumerate(self.voronoi.points):
			if self.areas[index] == 0:
				continue
			polygon = self.voronoi.regions[self.voronoi.point_region[index]]
			if len(polygon) < 3 or -1 in polygon:
				continue
			polygon = np.append(polygon, polygon[0])
#			self.lines_plot = self.ax.plot(points[polygon,0],
#										   points[polygon,1],
#									color = 'white',
#									linestyle = '-',
#									linewidth = 0.3,
#									zorder = 8)
			area_color = cmap(norm(self.areas[index]))
			self.areas_plot.append(self.ax.fill(points[polygon,0],
												points[polygon,1],
												color = area_color,
												linewidth = 0.,
												alpha = 1.,
												zorder = 8))
	
	def clear_canvas (self):
		self.remove_plot_element(self.image_plot)
		self.image_plot = None
		self.remove_plot_element(self.points_plot)
		self.points_plot = None
		self.remove_plot_element(self.lines_plot)
		self.lines_plot = None
		self.remove_plot_element(self.areas_plot)
		self.areas_plot = None
		xmin, xmax = self.ax.get_xlim()
		ymin, ymax = self.ax.get_ylim()
		self.ax.clear()
		self.ax.set_xlim([xmin,xmax])
		self.ax.set_ylim([ymin,ymax])
		self.ax.set_facecolor('black')
		self.draw()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for thingy in plot_element:
					try:
						self.remove_plot_element(thingy)
					except:
						print('problem')
			else:
				try:
					plot_element.remove()
				except:
					print('problem')
	
	def refresh (self):
	#	self.clear_canvas()
		self.plot_image()
		self.plot_points()
		self.plot_voronoi()
		self.plot_path()
		self.draw()
	
	def update_image (self, image = None):
		self.image = image
		self.image_changed = True
		self.refresh()
	
	def update_points (self, points = None):
		self.points = points
		self.points_changed = True
		self.refresh()
	
	def update_voronoi (self, voronoi = None, areas = None):
		self.voronoi = voronoi
		self.areas = areas
		self.voronoi_changed = True
		self.refresh()
	
	def update_switches (self, show_image = True, show_points = True,
							show_voronoi = True):
		if show_image != self.show_image:
			self.image_changed = True
			self.show_image = show_image
		if show_points != self.show_points:
			self.points_changed = True
			self.show_points = show_points
		if show_voronoi != self.show_voronoi:
			self.voronoi_changed = True
			self.show_voronoi = show_voronoi
		self.refresh()
	
	def update_path (self, path_vertices = np.zeros((0,2), dtype=int)):
		if len(path_vertices) == 0:
			self.path = None
		else:
			self.path = path_vertices
		self.path_changed = True
		self.refresh()
	
	def reset_zoom (self):
		if self.image is None:
			return False
		self.ax.set_ylim([self.image.shape[0]-1,0])
		self.ax.set_xlim([0,self.image.shape[1]-1])
		self.draw()
	
	def reset (self):
		self.clear_canvas()
		self.image = None
		self.points = None
		self.voronoi = None
		self.areas = None
		self.path = None
		self.image_plot = None
		self.points_plot = None
		self.lines_plot = None
		self.areas_plot = None
		self.path_plot = None
		self.show_image = True
		self.show_points = True
		self.show_voronoi = True
		self.show_path = True

################################################################################
# main window #
###############

class Window(QWidget):
	def __init__ (self):
		super().__init__()
		self.title = "Cell Density Tool"
		self.canvas = MPLCanvas()
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.canvas = MPLCanvas()
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.setWindowTitle(self.title)
		self.file_path = None
		self.image = None
		self.points = None
		self.voronoi = None
		self.areas = None
		self.densities = None
		self.path_vertices = np.zeros((0,2), dtype=int)
		self.frame = 0
		self.channel = 0
		self.neighbourhood_size = 16
		self.gauss_deviation = 2
		self.threshold_difference = 4
		self.area_threashold = 5000
		self.path_distance = 64
		self.path_number = 1000
		self.path_changed = False
		self.binning_number = 24
		# layout for full window
		main_layout = QVBoxLayout()
		main_layout.addWidget(self.canvas)
		toolbar_layout = QHBoxLayout()
		toolbar_layout.addWidget(self.toolbar)
		self.button_zoom = setup_button(self.reset_zoom,
										toolbar_layout,
										'Reset Zoom')
		self.checkbox_image = setup_checkbox(self.checkboxes,
											toolbar_layout,
											'show image',
											True)
		self.checkbox_points = setup_checkbox(self.checkboxes,
											toolbar_layout,
											'show points',
											True)
		self.checkbox_voronoi = setup_checkbox(self.checkboxes,
											toolbar_layout,
											'show voronoi',
											True)
		main_layout.addLayout(toolbar_layout)
		file_box, self.file_text = setup_labelbox(
						'<font color="red">File Name: </font>',
						'No file opened.')
		main_layout.addWidget(file_box)
		upper_layout = QHBoxLayout()
		self.button_open = setup_button(self.open_file,
											upper_layout,
											'Open File')
		self.frame_box = setup_combobox(
							self.select_frame,
							upper_layout, 'Frame:')
		self.channel_box = setup_combobox(
							self.select_channel,
							upper_layout, 'Channel:')
		number_box, self.number_text = setup_labelbox(
						'<font color="red">Number: </font>',
						'None')
		upper_layout.addWidget(number_box)
		self.button_draw_path = setup_button(self.draw_path,
										upper_layout,
										'Draw Path',
										toggle = True)
		self.textbox_distance = setup_textbox(self.get_textboxes,
											upper_layout,
											'Distance:')
		main_layout.addLayout(upper_layout)
		lower_layout = QHBoxLayout()
		self.button_find = setup_button(self.find_cells,
											lower_layout,
											'Find Cells')
		self.button_density = setup_button(self.calc_density,
											lower_layout,
											'Find Density')
		self.textbox_size = setup_textbox(self.get_textboxes,
											lower_layout,
											'Size:')
		self.textbox_sigma = setup_textbox(self.get_textboxes,
											lower_layout,
											'Deviation:')
		self.textbox_diff = setup_textbox(self.get_textboxes,
											lower_layout,
											'Difference:')
		self.textbox_area = setup_textbox(self.get_textboxes,
											lower_layout,
											'Max Area:')
		self.button_plot_densities = setup_button(self.plot_densities,
											lower_layout,
											'Plot density')
		main_layout.addLayout(lower_layout)
		self.setup_textboxes()
		self.setLayout(main_layout)
	
	def setup_textboxes (self):
		self.textbox_size.setText(str(self.neighbourhood_size))
		self.textbox_sigma.setText(str(self.gauss_deviation))
		self.textbox_diff.setText(str(self.threshold_difference))
		self.textbox_area.setText(str(self.area_threashold))
		self.textbox_distance.setText(str(self.path_distance))
	
	def reset_zoom (self):
		self.canvas.reset_zoom()
	
	def checkboxes (self):
		self.canvas.update_switches(
						show_image = self.checkbox_image.isChecked(),
						show_points = self.checkbox_points.isChecked(),
						show_voronoi = self.checkbox_voronoi.isChecked())
	
	def get_textboxes (self):
				self.neighbourhood_size = get_textbox(self.textbox_size,
											minimum_value = 1,
											maximum_value = 128,
											is_int = True)
				self.gauss_deviation = get_textbox(self.textbox_sigma,
											minimum_value = 0,
											maximum_value = 16,
											is_int = True)
				self.threshold_difference = get_textbox(self.textbox_diff,
											minimum_value = 0,
											maximum_value = 16,
											is_int = True)
				self.area_threashold = get_textbox(self.textbox_area,
											minimum_value = 0,
											maximum_value = 12000,
											is_int = True)
				self.path_distance = get_textbox(self.textbox_distance,
											minimum_value = 0,
											maximum_value = 12000,
											is_int = True)
	
	def select_frame (self):
		self.frame = self.frame_box.currentIndex()
		self.update_image()
	
	def select_channel (self):
		self.channel = self.channel_box.currentIndex()
		self.update_image()
	
	def file_dialog (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								'Open Microscope File', '',
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.tif' or \
			   file_path.suffix.lower() == '.tiff':
				self.file_path = file_path
				return True
			else:
				self.file_path = None
				return False
	
	def open_file (self):
		self.file_path = None
		self.image = None
		self.points = None
		self.voronoi = None
		self.areas = None
		self.densities = None
		self.frame_box.clear()
		self.frame = 0
		self.channel_box.clear()
		self.channel = 0
		self.canvas.reset()
		self.file_text.setText('No file opened.')
		if self.file_dialog():
			self.file_text.setText(str(self.file_path))
			if self.file_path.suffix.lower() == '.tif' or \
					self.file_path.suffix.lower() == '.tiff':
				self.image = read_tiff(self.file_path)
				if len(self.image.shape) == 3:
					self.image = self.image[:,:,:,np.newaxis]
					self.channel_box.addItem('0')
				else:
					for index in range(self.image.shape[3]):
						self.channel_box.addItem(f'{index:d}')
				self.channel_box.setCurrentIndex = 0
				for index in range(self.image.shape[0]):
					self.frame_box.addItem(f'{index:d}')
				self.frame_box.setCurrentIndex = 0
				print(self.image.shape)
			self.update_image()
	
	def update_image(self):
		if self.image is None:
			return False
		self.canvas.update_image(self.image[self.frame,:,:, self.channel])
		self.canvas.reset_zoom()
	
	def find_cells (self):
		if self.image is None:
			return False
		self.points = find_centres(self.image[self.channel,:,:],
						neighbourhood_size = self.neighbourhood_size,
						threshold_difference = self.threshold_difference,
						gauss_deviation = self.gauss_deviation)
		self.number_text.setText(str(len(self.points)))
		self.canvas.update_points(self.points)
	
	def calc_density (self):
		if self.points is None:
			return False
		self.voronoi = Voronoi(self.points)
		self.areas = np.zeros(self.points.shape[0])
		for index, point in enumerate(self.points):
			polygon = self.voronoi.regions[self.voronoi.point_region[index]]
			if len(polygon) < 3 or -1 in polygon:
				continue
			vectors = self.voronoi.vertices[polygon] - point[np.newaxis,:]
			lengths = np.linalg.norm(vectors, axis=1)
			if np.amax(lengths) > 8*np.amin(lengths) or \
			   np.amax(lengths) > 3*np.median(lengths):
				continue
			self.areas[index] = PolyArea(self.voronoi.vertices[polygon,0],
										 self.voronoi.vertices[polygon,1])
		self.areas[self.areas>self.area_threashold] = 0
		self.canvas.update_voronoi(self.voronoi, self.areas)
	
	def plot_densities (self):
		if self.densities is None:
			return
		plt.plot(self.densities[:,0],
				 self.densities[:,1],
					color = 'tab:blue',
					linestyle = '',
					marker = '.',
					zorder = 6)
		bins = np.linspace(0,1, self.binning_number)
		indices = np.digitize(self.densities[:,0], bins)
		indices -= 1
		binned = np.zeros((len(np.unique(indices)), 2), dtype = float)
		for index in np.unique(indices):
			if len(self.densities[indices==index,0]) == 0:
				continue
			binned[index,0] = np.mean(self.densities[indices==index,0])
			binned[index,1] = np.mean(self.densities[indices==index,1])
		plt.plot(binned[:,0], binned[:,1],
					color = 'tab:orange',
					linestyle = '-',
					marker = '',
					zorder = 7)
		plt.ylabel('Density (#/pixel^2)')
		plt.xlabel('Position Along Path (Normalized)')
		plt.show()
	
	def map_densities (self):
		if not self.path_changed:
			return False
		if self.points is None or self.areas is None:
			return False
		if len(self.path_vertices) < self.path_number:
			return False
		tree = KDTree(self.points)
		included = tree.query_ball_point(self.path_vertices,
											self.path_distance)
		included = np.unique(np.concatenate(included, axis=None))
		self.densities = np.zeros((len(included),2), dtype = float)
		points = self.points[included]
		print(points.shape)
		print(self.densities.shape)
		areas = self.areas[included]
		for index in np.arange(points.shape[0]):
			closest = np.argmin(np.linalg.norm(
										self.path_vertices - points[index],
																	axis=1))
			self.densities[index,0] = closest
			self.densities[index,1] = 1/areas[index]
		self.densities[:,0] /= self.path_number
		self.path_changed = False
		self.plot_densities()
	
	def draw_path (self):
		self.drawing = self.button_draw_path.isChecked()
		if self.drawing:
			self.path_vertices = np.zeros((0,2), dtype = int)
			self.canvas.update_path(self.path_vertices)
			self.click_id = self.canvas.mpl_connect('button_press_event',
															self.on_click)
		else:
			if len(self.path_vertices) < 2:
				self.path_vertices = np.zeros((0,2), dtype = int)
				return False
			self.canvas.mpl_disconnect(self.click_id)
			spline = make_interp_spline(np.arange(len(self.path_vertices)) / \
											(len(self.path_vertices)-1),
										self.path_vertices)
			t = np.linspace(0,1,self.path_number)
			self.path_vertices = spline(t)
			self.path_changed = True
			self.canvas.update_path(self.path_vertices)
			self.map_densities()
	
	def on_click (self, event):
		if self.image is None:
			return False
		self.position = np.array([int(np.floor(event.xdata)),
								  int(np.floor(event.ydata))])
		if (self.position[0] < 0) or \
		   (self.position[0] > self.image.shape[1]) or \
		   (self.position[1] < 0) or \
		   (self.position[1] > self.image.shape[2]):
			return False
		if event.button is MouseButton.LEFT:
			self.path_vertices = np.append(self.path_vertices,
										self.position[np.newaxis,:], axis=0)
			self.canvas.update_path(self.path_vertices)
		elif event.button is MouseButton.RIGHT:
			self.button_draw_path.setChecked(False)
			self.draw_path()

################################################################################

if __name__ == "__main__":
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())


