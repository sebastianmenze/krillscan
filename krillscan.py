# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:01:11 2023

@author: Administrator
"""


from skimage.transform import  resize
from skimage.transform import  resize_local_mean
import shutil
from skimage.transform import  resize

from krillscan.echolab2.instruments import EK80, EK60
import configparser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os

# from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import convolve2d
# from skimage.transform import  resize

from krillscan.echopy import transform as tf
from krillscan.echopy import resample as rs
from krillscan.echopy import mask_impulse as mIN
from krillscan.echopy import mask_seabed as mSB
from krillscan.echopy import get_background as gBN
from krillscan.echopy import mask_range as mRG
from krillscan.echopy import mask_shoals as mSH
from krillscan.echopy import mask_signal2noise as mSN

from pyproj import Geod
geod = Geod(ellps="WGS84")
from pathlib import Path


# from matplotlib.colors import ListedColormap
import re
import traceback
# from pyproj import Proj, transform
import zipfile

import smtplib
import ssl
# import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.text  import MIMEText

from threading import Timer

import xarray as xr
from scipy.interpolate import interp1d
from scipy import integrate



# from echolab2.instruments import EK80, EK60
import configparser
from scipy.signal import find_peaks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os
import logging

from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import convolve2d
from scipy.interpolate import interp1d


from skimage.transform import rescale, resize 

from pyproj import Geod
geod = Geod(ellps="WGS84")

import sys
import matplotlib

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import qdarktheme
import time

from matplotlib.path import Path as MPL_Path

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.colors import ListedColormap
import re
import traceback
import zipfile

import xarray as xr
import matplotlib.dates as mdates
#%% inspection GUI

class MplCanvas(FigureCanvasQTAgg ):

    def __init__(self, parent=None, dpi=150):
        self.fig = Figure(figsize=None, dpi=dpi,facecolor='gray')
        super(MplCanvas, self).__init__(self.fig)

        
class MainWindow(QtWidgets.QMainWindow):
    

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.canvas =  MplCanvas(self, dpi=150)
                
        self.echodata=dict()
        self.echodata_swarm=dict()
        self.df_nasc=pd.DataFrame([])

        
        self.filecounter=-1
        self.filenames = None
        self.df_files = pd.DataFrame([])

        self.folder_source=''
        self.statusBar().setStyleSheet("background-color : k")


        menuBar = self.menuBar()

        # Creating menus using a title
        openMenu = menuBar.addAction("Select echogram files")
        openMenu.triggered.connect(self.openfiles_fun)

        
        self.showfolderbutton =  menuBar.addAction('Show data folder')
        # self.showfolderbutton.setEnabled(False)
        self.showfolderbutton.triggered.connect(self.showfoldefunc)     

        
        self.showfolderbutton =  menuBar.addAction('Undo mask change')
        # self.showfolderbutton.setEnabled(False)
        self.showfolderbutton.triggered.connect(self.undo_mask)     
  
        quitMenu = menuBar.addAction("Quit")
        quitMenu.triggered.connect(self.func_quit)     
    

        toolbar = QtWidgets.QToolBar()

        self.checkbox_mask=QtWidgets.QCheckBox('Save changes')
        self.checkbox_mask.setChecked(False)            
        toolbar.addWidget(self.checkbox_mask)
        
        self.checkbox_raw=QtWidgets.QCheckBox('Raw')
        self.checkbox_raw.setChecked(False)           
        self.checkbox_raw.clicked.connect(self.choose_raw_changed)          
        toolbar.addWidget(self.checkbox_raw)

        toolbar.addWidget(QtWidgets.QLabel('Freq.:'))        
        self.choose_freq = QtWidgets.QComboBox()
        self.choose_freq.addItem('38')
        self.choose_freq.addItem('120')
        self.choose_freq.addItem('200')
        self.choose_freq.setCurrentIndex(1)
        toolbar.addWidget(  self.choose_freq)
        toolbar.addWidget(QtWidgets.QLabel('kHz'))            
        self.choose_freq.activated[str].connect(self.choose_freq_changed) 

        toolbar.addWidget(QtWidgets.QLabel('Color:'))        
        self.choose_color = QtWidgets.QComboBox()
        self.choose_color.addItem('viridis')
        self.choose_color.addItem('plasma')
        self.choose_color.addItem('jet')
        self.choose_color.setCurrentIndex(2)
        toolbar.addWidget(  self.choose_color)
        self.choose_color.activated[str].connect(self.choose_color_changed) 
        
        

        toolbar.addWidget(QtWidgets.QLabel('Min. depth:'))        
        self.excl_depth_spin = QtWidgets.QSpinBox()
        self.excl_depth_spin.setMinimum(0)
        self.excl_depth_spin.setMaximum(8000)
        self.excl_depth_spin.setValue(20)
        toolbar.addWidget(  self.excl_depth_spin)
        toolbar.addWidget(QtWidgets.QLabel('m'))        

        toolbar.addWidget(QtWidgets.QLabel('Max. depth:'))        
        self.max_depth_spin = QtWidgets.QSpinBox()
        self.max_depth_spin.setMinimum(0)
        self.max_depth_spin.setMaximum(8000)
        self.max_depth_spin.setValue(500)

        toolbar.addWidget(  self.max_depth_spin)
        toolbar.addWidget(QtWidgets.QLabel('m'))              
        
        self.butt_prev=QtWidgets.QPushButton('<-- previous')
        self.butt_prev.clicked.connect(self.plot_prev)        
        toolbar.addWidget(self.butt_prev)

        self.butt_next=QtWidgets.QPushButton('Next -->')
        self.butt_next.clicked.connect(self.plot_next)        
        toolbar.addWidget(self.butt_next)
        
                #### hotkeys
        self.msgSc1 = QtWidgets.QShortcut(QtCore.Qt.Key_Right, self)
        self.msgSc1.activated.connect(self.plot_next)
        self.msgSc2 = QtWidgets.QShortcut(QtCore.Qt.Key_Left, self)
        self.msgSc2.activated.connect(self.plot_prev)        


        self.butt_removearea=QtWidgets.QPushButton('Remove area')
        self.butt_removearea.clicked.connect(self.mask_removearea)        
        toolbar.addWidget(self.butt_removearea)
        
        self.butt_addarea=QtWidgets.QPushButton('Add area')
        self.butt_addarea.clicked.connect(self.mask_addarea)        
        toolbar.addWidget(self.butt_addarea)
        
        toolbar.addSeparator()

         
        tnav = NavigationToolbar( self.canvas, self)       
        toolbar.addWidget(tnav)
       
        outer_layout = QtWidgets.QVBoxLayout()
        outer_layout.addWidget(toolbar)
        outer_layout.addWidget(self.canvas)
    
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)
        
        
        self.show()  
   

        self.cmap_mask = ListedColormap( np.array([[1,0,0, 1. ]] ) )
        
        
        self.t=[-1,-1]
        self.plotwindow_startsecond=0
        self.plotwindow_length=0
  
        
    def settings_edit(self):
        os.startfile(self.ini_file)    
   
                  
    def showfoldefunc(self):    
         os.startfile(self.workpath)
         
    def openfiles_fun(self):
        
        fname_canidates, ok = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()",'',"Echogram Files (*_echogram.nc)")
         
        if len( fname_canidates ) >0:
            
            self.filenames = np.array( fname_canidates )   
            print(  fname_canidates )
            
            self.workpath = os.path.dirname( fname_canidates[0])
        
            self.filecounter=-1
               
            self.plot_next()
            
    def choose_color_changed(self):
        if self.filecounter>=0:        
            self.plot_echogram()

    def choose_raw_changed(self):
        if self.filecounter>=0:        
            self.plot_echogram()
            
    def plot_echogram(self):    
        
                
        # breakpoint()
        
        ix_f = self.xr_sv.coords['frequency']== int(self.choose_freq.currentText()) *1000
        sv = np.squeeze( self.xr_sv[ix_f,:,:])
        

            
                
        # sv = sv.interpolate_na( dim='time', method="linear")
        
        # sv = sv.interpolate_na(  'depth',max_gap=3,method="linear")
        # sv = sv.interpolate_na(  'time' ,max_gap=pd.Timedelta(seconds=5))
        
        # sv = sv.interp(time=sv.time , depth=sv.depth, method="linear")
        
        

        sv_mask=sv.copy()
        sv_mask.values [~self.mask_manual.values]=np.nan

        
        cell_thickness=np.abs(np.mean(np.diff( self.xr_sv.coords['depth']) ))        
        
        track= pd.DataFrame([])
        track.index= self.xr_sv.coords['time']      
        track['nasc_swarm'] =4*np.pi*1852**2 * np.nansum( np.power(10, sv_mask /10)*cell_thickness ,axis=0)   
          
        # peak filter
        x = track['nasc_swarm'].values.copy()
        peaks, peak_meta = find_peaks(x,prominence=10000,width=[0,2] )
        # x[peaks]=np.nan
        
        ix1 = np.round( peak_meta['left_ips']).astype(int)-1
        ix2 = np.round( peak_meta['right_ips']).astype(int)+1
        
        ix1[ix1<0]=0
        ix2[ix2>len(x)-1]=len(x)-1
        
        for i1,i2 in zip(ix1,ix2):
            x[i1:i2]=np.nan

        track['nasc_swarm']=x
        track['nasc_swarm']=track['nasc_swarm'].interpolate()
        
        # duration =     ( self.xr_sv.coords['time'][-1] - self.xr_sv.coords['time'][0]).data  / np.timedelta64(1, 's')
        # dm=self.xr_sv.coords['depth'][-1].data
        
        
        self.canvas.fig.clf() 
                
        self.ax = self.canvas.fig.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]},sharex=True)
        ax=self.ax
        
        ax[0].plot( track['nasc_swarm'] , '-k')
        ax[0].plot( track['nasc_swarm'].resample('1min').mean(), '.-b')
        
        
        ax[0].grid()
        # ax[0].set_xlim( self.xr_sv.coords['time'].min() , self.xr_sv.coords['time'].max())
        ax[0].set_ylabel('NASC')
        
        # plt.gca=ax[1]
        
        x_lims = mdates.date2num( [  self.xr_sv.coords['time'].min().values , self.xr_sv.coords['time'].max().values ])
        range_max = self.xr_sv.coords['depth'].max()
        ext=[x_lims[0] ,x_lims[1],-range_max ,0 ]
        
        if self.checkbox_raw.isChecked():
            fname= self.filenames[self.filecounter]
            self.xr_sv_raw = xr.open_dataarray(fname[:-12] + '_rawechogram.nc')
            sv_raw = np.squeeze( self.xr_sv_raw[ix_f,:,:])
            im = ax[1].imshow( sv_raw,cmap= self.choose_color.currentText()   ,aspect='auto',vmin=-90,vmax=-30,extent=ext)
            
        else:    
            im = ax[1].imshow( sv,cmap= self.choose_color.currentText()   ,aspect='auto',vmin=-90,vmax=-30,extent=ext)
        
        masked_data = np.ma.masked_where(self.mask_manual == 1, self.mask_manual)
        
        
        ax[1].imshow( masked_data  ,aspect='auto',extent=ext,cmap=self.cmap_mask,alpha=0.3)
        
        # cbar3 = plt.colorbar(im)
        ax[1].xaxis_date()
        ax[1].grid()
        ax[1].set_ylabel('Depth in m')
        ax[1].set_ylim(-self.max_depth_spin.value(),0)

        ax[1].plot(x_lims,[-self.excl_depth_spin.value(),-self.excl_depth_spin.value()] ,'-r' )
        
        

        cbaxes = inset_axes(ax[1], width="15%", height="3%", loc='lower left',  bbox_to_anchor=(0.05, 0.15, 1, 1) ,bbox_transform= ax[1].transAxes) 
        self.canvas.fig.colorbar(im,cax=cbaxes,label='$s_v$', orientation='horizontal')        
            
        
            #         cbaxes = inset_axes(self.canvas.axes4, width="100%", height="100%", loc=3,bbox_to_anchor=(.05, .15, .4, .04),bbox_transform=self.canvas.axes4.transAxes) 
            # cbar=self.canvas.fig.colorbar(sc,cax=cbaxes, orientation='horizontal')
            
            

        self.canvas.fig.tight_layout()
        self.canvas.draw()

   
    def choose_freq_changed(self):
        if self.filecounter>=0:   
            self.plot_echogram()

        
    def read_file_and_mask(self):    
        if self.filecounter>=0:        
            fname= self.filenames[self.filecounter]
            
            
            self.xr_sv = xr.open_dataarray(fname)
            
            # breakpoint()
            
            f_list = (self.xr_sv.coords['frequency'].values / 1000).astype(int).astype(str)
            self.choose_freq.clear()
            self.choose_freq.addItems(f_list)
            
            try:
                ix_120 = np.where( self.xr_sv.coords['frequency'].values==120000)[0][0] 
            except:
                ix_120 = np.argmax( self.xr_sv.coords['frequency'].values)
                
            
            self.choose_freq.setCurrentIndex( ix_120 )
            # self.choose_freq.setCurrentIndex( np.argmax(self.xr_sv.coords['frequency'].values) )

            if os.path.isfile(fname[:-12] + '_mask_manual.h5' ):     
                self.mask_manual= np.transpose(  pd.read_hdf( fname[:-12] + '_mask_manual.h5',key='df' ) )
            else:
                if os.path.isfile(fname[:-12] + '_mask_swarm.h5' ):     
                    self.mask_swarm= np.transpose(  pd.read_hdf( fname[:-12] + '_mask_swarm.h5',key='df' ) )
                    self.mask_manual = self.mask_swarm.copy()
                    ixdepthvalid= ( self.xr_sv.coords['depth'] >= self.excl_depth_spin.value() ) & ( self.xr_sv.coords['depth']  <= self.max_depth_spin.value())
                    self.mask_manual.values[~ixdepthvalid,:]=False  
                else:
                    
                    s2=len( self.xr_sv.coords['time'])
                    s1=len( self.xr_sv.coords['depth'])
                    self.mask_manual=pd.DataFrame( np.ones([s1,s2]).astype(bool)   )
                    self.mask_manual.columns=self.xr_sv.coords['time']
                    self.mask_manual.index=self.xr_sv.coords['depth']
                    ixdepthvalid= ( self.xr_sv.coords['depth'] >= self.excl_depth_spin.value() ) & ( self.xr_sv.coords['depth']  <= self.max_depth_spin.value())
                    self.mask_manual.values[~ixdepthvalid,:]=False                  
          
            self.mask_manual_old=self.mask_manual.copy() 
  
                
            # self.mask_dbdiff= np.transpose( pd.read_hdf( fname[:-12] + '_mask_dbdiff.h5',key='df' ) )
            

            
    def undo_mask(self):
        self.mask_manual=self.mask_manual_old.copy() 
        self.plot_echogram()            

                
    def onclick_draw(self,event):
            # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #       ('double' if event.dblclick else 'single', event.button,
            #        event.x, event.y, event.xdata, event.ydata))
            if event.button==1 & event.dblclick:
                self.draw_x=self.draw_x.append( pd.Series(event.xdata) ,ignore_index=True )
                self.draw_y=self.draw_y.append( pd.Series(event.ydata) ,ignore_index=True )
                # self.f_limits=self.canvas.axes.get_ylim()
                # self.t_limits=self.canvas.axes.get_xlim()   
                
                line = self.line_2.pop(0)
                line.remove()        
                self.line_2 =self.ax[1].plot(self.draw_x,self.draw_y,'.-r')      
                self.canvas.draw()    
                         
                # func_draw_shape_plot()   
              
            if event.button==3:
                self.draw_x=self.draw_x.head(-1)
                self.draw_y=self.draw_y.head(-1)
                # self.f_limits=self.canvas.axes.get_ylim()
                # self.t_limits=self.canvas.axes.get_xlim()
                # func_draw_shape_plot()              
                line = self.line_2.pop(0)
                line.remove()        
                self.line_2 =self.ax[1].plot(self.draw_x,self.draw_y,'.-r')     
                self.canvas.draw()                   
 
    def mask_removearea(self):
            
        # msg = QtWidgets.QMessageBox()
        # msg.setIcon(QtWidgets.QMessageBox.Information)   
        # msg.setText("Add points with double left click.\nRemove latest point with single right click. \nExit draw mode by pushing enter")
        # msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        # returnValue = msg.exec()    
        # if returnValue == QtWidgets.QMessageBox.Ok:
            self.butt_removearea.setEnabled(False)   
            print('drawing')  
            self.draw_x=pd.Series(dtype='float')
            self.draw_y=pd.Series(dtype='float')
            self.d_limits=self.ax[1].get_ylim()
            self.t_limits=self.ax[1].get_xlim()
            self.plot_echogram()
            
            # self.canvas.fig.canvas.mpl_disconnect(self.cid1)    
            self.cid2=self.canvas.fig.canvas.mpl_connect('button_press_event', self.onclick_draw)
            self.line_2 =self.ax[1].plot(self.draw_x,self.draw_y,'.-r')        
            # self.plot_echogram()   
            self.drawexitm = QtWidgets.QShortcut(QtCore.Qt.Key_Return, self)
            self.drawexitm.activated.connect(self.func_draw_shape_exit_remove)  
         
    def func_draw_shape_exit_remove(self):
        # print('save shape' + str(self.draw_x.shape))
        self.canvas.fig.canvas.mpl_disconnect(self.cid2)
        ## deactive shortcut
        self.drawexitm.setEnabled(False)  

        if self.draw_x.shape[0]>2:

            
            # breakpoint()
            
            m = np.transpose( self.mask_manual )
            t = mdates.date2num( m.index )
            d = -m.columns 
            

            kk_t,kk_d=np.meshgrid( t,d)   
            # kernel=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] )
            x, y = kk_t.flatten(), kk_d.flatten()
            points = np.vstack((x,y)).T 
            
            p = MPL_Path(list(zip( self.draw_x.values , self.draw_y.values))) # make a polygon
            grid = p.contains_points(points)
            m_shapemask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
            self.mask_manual_old=self.mask_manual.copy() 
            self.mask_manual.values[ m_shapemask ] = False
                  
        self.plot_echogram()
        self.butt_removearea.setEnabled(True)   

         
    def func_draw_shape_exit_add(self):
        # print('save shape' + str(self.draw_x.shape))
        self.canvas.fig.canvas.mpl_disconnect(self.cid2)
        ## deactive shortcut
        self.drawexitm.setEnabled(False)  

        if self.draw_x.shape[0]>2:

            # breakpoint()
            
            m = np.transpose( self.mask_manual )
            t = mdates.date2num( m.index )
            d = -m.columns 
            

            kk_t,kk_d=np.meshgrid( t,d)   
            # kernel=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] )
            x, y = kk_t.flatten(), kk_d.flatten()
            points = np.vstack((x,y)).T 
            
            p = MPL_Path(list(zip( self.draw_x.values , self.draw_y.values))) # make a polygon
            grid = p.contains_points(points)
            m_shapemask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
            self.mask_manual_old=self.mask_manual.copy() 
            self.mask_manual.values[ m_shapemask ] = True
                  
        self.plot_echogram()   
        self.butt_addarea.setEnabled(True)     
         


    def mask_addarea(self):
        # msg = QtWidgets.QMessageBox()
        # msg.setIcon(QtWidgets.QMessageBox.Information)   
        # msg.setText("Add points with double left click.\nRemove latest point with single right click. \nExit draw mode by pushing enter")
        # msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        # returnValue = msg.exec()    
        # if returnValue == QtWidgets.QMessageBox.Ok:
            self.butt_addarea.setEnabled(False)     

            print('drawing')  
            self.draw_x=pd.Series(dtype='float')
            self.draw_y=pd.Series(dtype='float')
            self.d_limits=self.ax[1].get_ylim()
            self.t_limits=self.ax[1].get_xlim()
            self.plot_echogram()
            
            # self.canvas.fig.canvas.mpl_disconnect(self.cid1)    
            self.cid2=self.canvas.fig.canvas.mpl_connect('button_press_event', self.onclick_draw)
            self.line_2 =self.ax[1].plot(self.draw_x,self.draw_y,'.-g')        
            # self.plot_echogram()   
            self.drawexitm = QtWidgets.QShortcut(QtCore.Qt.Key_Return, self)
            self.drawexitm.activated.connect(self.func_draw_shape_exit_add)  
        
        
        
    def save_changes(self):
       
        
        m = np.transpose( self.mask_manual )
        fname= self.filenames[self.filecounter]
        m.to_hdf( fname[:-12] + '_mask_manual.h5',key='df' )
        
        # calc nasc
        ix_f = self.xr_sv.coords['frequency']== int(self.choose_freq.currentText()) *1000
        sv_mask = np.squeeze( self.xr_sv[ix_f,:,:]).copy()
        sv_mask.values [~self.mask_manual.values]=np.nan

        cell_thickness=np.abs(np.mean(np.diff( self.xr_sv.coords['depth']) ))        
        nasc = 4*np.pi*1852**2 * np.nansum( np.power(10, sv_mask /10)*cell_thickness ,axis=0)   

        # peak filter
        x = nasc.copy()
        peaks, peak_meta = find_peaks(x,prominence=10000,width=[0,2] )
        # x[peaks]=np.nan
       
        ix1 = np.floor( peak_meta['left_ips']).astype(int)-1
        ix2 = np.ceil( peak_meta['right_ips']).astype(int)+1         
        ix1[ix1<0]=0
        ix2[ix2>len(x)-1]=len(x)-1
         
        for i1,i2 in zip(ix1,ix2):
             x[i1:i2]=np.nan
        
        track=  pd.read_hdf( fname[:-12] + '_nasctable.h5',key='df' ) 

        track['nasc_manual']=x
        track['nasc_manual']=track['nasc_manual'].interpolate()


        # track['nasc_manual'] =nasc
        track.to_hdf( fname[:-12] + '_nasctable.h5',key='df'  )
        
        
        
    def plot_next(self):
         if len(self.filenames)>0:
            print('old filecounter is: '+str(self.filecounter))
            
            if self.checkbox_mask.isChecked():
                self.save_changes()
                
           
            self.filecounter=self.filecounter+1
            if self.filecounter>len(self.filenames)-1:
                self.filecounter=len(self.filenames)-1
                print('That was it')
            self.read_file_and_mask()
            self.plot_echogram()
            
            

    def plot_prev(self):
         if len(self.filenames)>0:   
            print('old filecounter is: '+str(self.filecounter))
            
            if self.checkbox_mask.isChecked():
                self.save_changes()

         
            self.filecounter=self.filecounter-1
            if self.filecounter<0:
                self.filecounter=0
                print('That was it')
            # new file    
            # self.filecounter=self.filecounter+1
            self.read_file_and_mask()
            self.plot_echogram()
                
         
     
######
             

    def func_quit(self):
        self.statusBar().setStyleSheet("background-color : k")
        # self.statusBar().removeWidget(self.label_1)   
        # self.startautoMenu.setEnabled(True)
        # self.exitautoMenu.setEnabled(False)     
        QtWidgets.QApplication.instance().quit()     
        # QCoreApplication.quit()
        self.close()    
        




#%% automatic processing


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

class krillscan_class():
    
    def inspect(self):
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("Krillscan")    
        app.setStyleSheet(qdarktheme.load_stylesheet())
        w = MainWindow()
        sys.exit(app.exec_())
             
    
    def start(self,ini_file):
        print('start')
        self.ini_file = ini_file
        self.callback_process_active=False
        self.callback_email_active=False
        # self.callback_plot_active=False
        self.df_files=pd.DataFrame([])
        # self.echogram=pd.DataFrame([])    
        self.positions=pd.DataFrame([])    
        
        config = configparser.ConfigParser()
        config.read(self.ini_file)            
        self.workpath=  str(config['GENERAL']['target_folder'])  
        
        print('work dir: ' +os.getcwd())
        print('source dir: ' + str(config['GENERAL']['source_folder'])  )
        print('target dir: ' +self.workpath)
               

        self.timer_process = RepeatTimer(10, self.callback_process_raw)
        self.timer_process.start()
        self.timer_email = RepeatTimer(30, self.callback_email)
        self.timer_email.start()  
        
        # doc.add_periodic_callback( self.callback_process_raw,1000 ) 
        # doc.add_periodic_callback( self.callback_plot,3000 ) 
        # doc.add_periodic_callback( self.callback_email,5000 ) 

                
    def stop(self):
        self.timer_process.cancel()
        self.timer_email.cancel()  
        print('Krillscan stopped')


    def read_raw(self,rawfile):       
        # df_sv=pd.DataFrame( [] )
        positions=pd.DataFrame( []  )
                
        # breakpoint()
        
        # print('Echsounder data are: ')
        self.config = configparser.ConfigParser()
        self.config.read(self.ini_file)   
        
        # rawfile=r"E:\KRILL-2023\D20230129-T065613.raw"
        # rawfile=r"./source_folder\D20230128-T105149.raw"
   
        try:     
            raw_obj = EK80.EK80()
            raw_obj.read_raw(rawfile)
            print(raw_obj)
        except Exception as e:            
            print(e)       
            try:     
                raw_obj = EK60.EK60()
                raw_obj.read_raw(rawfile)
                print(raw_obj)
            except Exception as e:
                print(e)       
                
                                           
        
        raw_freq= list(raw_obj.frequency_map.keys())
        
        
        # self.ekdata=dict()
        
        # for f in raw_freq:
        # f=float(self.config['GENERAL']['scrutinization_frequency'])
        print(raw_freq)
        
        for f in raw_freq:
            print(raw_obj.frequency_map[f])
            
            raw_data = raw_obj.raw_data[raw_obj.frequency_map[f][0]][0]  
    
            if np.shape(raw_data)[0]>1:                     
                cal_obj = raw_data.get_calibration()
                
                try: 
                   cal_obj.gain=float(self.config['CALIBRATION']['gain']       )
                except:
                    pass
                try: 
                   cal_obj.sa_correction=float(self.config['CALIBRATION']['sa_correction']       )
                except:
                    pass
                try: 
                   cal_obj.beam_width_alongship=float(self.config['CALIBRATION']['beam_width_alongship']       )
                except:
                    pass
                try: 
                   cal_obj.beam_width_athwartship=float(self.config['CALIBRATION']['beam_width_athwartship']       )
                except:
                    pass
                try: 
                   cal_obj.angle_offset_alongship=float(self.config['CALIBRATION']['angle_offset_alongship']       )
                except:
                    pass
                try: 
                   cal_obj.angle_offset_athwartship=float(self.config['CALIBRATION']['angle_offset_athwartship']       )
                except:
                    pass
                    
                
                sv_obj = raw_data.get_sv(calibration = cal_obj)    
                sv_obj = raw_data.get_sv()    
                  
                # positions =pd.DataFrame(  raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1] )
               
                svr = np.transpose( 10*np.log10( sv_obj.data ) )

                # print(sv_obj.range.max())
                # r=np.arange( sv_obj.range.min() , sv_obj.range.max() , 0.5 )
                r=np.arange( 0 , sv_obj.range.max() , 0.5 )
        
                t=sv_obj.ping_time
        
                sv=  resize(svr,[ len(r) , len(t) ] )
                
               # print(sv.shape)
               
               # Clean impulse noise      
                sv_im, m120in_ = mIN.wang(sv, thr=(-70,-40), erode=[(3,3)],
                                dilate=[(7,7)], median=[(7,7)])
                
                if f== float(self.config['GENERAL']['scrutinization_frequency'] ):
                    
                    positions =pd.DataFrame(  raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1] )

                    seafloor_sv_threshold_db = float(self.config['PROCESSING']['seafloor_sv_threshold_db'] )

                    mb = mSB.ariza(sv, r, r0=10, r1=1000, roff=0,
                                      thr=seafloor_sv_threshold_db, ec=1, ek=(3,3), dc=10, dk=(5,15))
                    
                    print(mb.sum())
                    # breakpoint()
                    # sv_im[mb]=-999
                    
                    bottomdepth=[]         
                    for j in range(mb.shape[1]):
                        row_1=mb[:,j]
                        if np.sum(row_1==True)>0:
                            bottomdepth.append( np.min(r[row_1==True]) )
                        else:
                            bottomdepth.append( r.max() )
                    # print(bottomdepth)   
                    positions['bottomdepth_m']=bottomdepth


                # estimate and correct background noise       
                p         = np.arange(len(t))                
                s         = np.arange(len(r))          
                bn, m120bn_ = gBN.derobertis(sv, s, p, 5, 20, r, np.mean(cal_obj.absorption_coefficient) ) # whats correct absoprtion?
                b=pd.DataFrame(bn)
                bn=  b.interpolate(axis=1).interpolate(axis=0).values                        
                sv_clean     = tf.log(tf.lin(sv_im) - tf.lin(bn))
              
                sv_SNR_threshold_db = float(self.config['PROCESSING']['sv_SNR_threshold_db'] )
                msn             = mSN.derobertis(sv_clean, bn, thr=sv_SNR_threshold_db)
                sv_clean[msn] = np.nan

              
                sv_x = xr.DataArray(sv_clean,coords={"time": t,"depth": r},  dims=("depth","time"))
                
                # fill holes
                # sv_x = sv_x.interpolate_na( dim='time', method="linear",max_gap=pd.Timedelta(seconds=2))
                sv_x = sv_x.interpolate_na( dim='depth', method="linear",max_gap=3)
               
                sv_x_raw = xr.DataArray(sv,coords={"time": t,"depth": r},  dims=("depth","time"))
                # breakpoint()

                
                if f==raw_freq[0]:
                    xr_sv=sv_x
                    xr_sv_raw=sv_x_raw
                else:
                    xr_sv = xr.concat([xr_sv,sv_x], dim="frequency")
                    xr_sv_raw = xr.concat([xr_sv_raw,sv_x_raw], dim="frequency")

                    
        xr_sv.coords['frequency'] = raw_freq
        xr_sv_raw.coords['frequency'] = raw_freq
        
        # duration =     ( xr_sv.coords['time'][-1] - xr_sv.coords['time'][0]).data  / np.timedelta64(1, 's')
        # dm=xr_sv.coords['depth'][-1].data
        
        # fig= plt.figure(0)
        # plt.clf()
        # fig.set_size_inches(10,10)
        # k=1
        # for f in xr_sv.coords['frequency']:
        #     if k==1:
        #         ax1=plt.subplot(len(xr_sv.coords['frequency']),1,k)
        #     else:
        #         plt.subplot(len(raw_freq),1,k,sharex=ax1,sharey=ax1)
        #     plt.imshow(xr_sv[k-1,:,:],aspect='auto',vmin=-90,vmax=-30,extent=[0,duration,dm,0])
        #     plt.title(str(xr_sv.coords['frequency'].data[k-1]))
        #     plt.colorbar()
        #     plt.grid()
        #     k=k+1
        # plt.tight_layout()
        
        # print(xr_sv)
        # print(positions)
               
        return xr_sv, positions , xr_sv_raw

    def callback_process_raw(self):
              
      config = configparser.ConfigParser()
      config.read(self.ini_file)            
      self.folder_source=  str(config['GENERAL']['source_folder'])  
        
      if (self.callback_process_active==False) :
          
        self.callback_process_active==True
        # self.workpath=  os.path.join(self.folder_source,'krill_data')     
        # os.chdir(self.workpath)
    
        new_df_files = pd.DataFrame([])   

        globstr =  os.path.join( glob.escape( self.folder_source),'*.raw')           
        new_df_files['path'] = glob.glob( globstr )  
        print('found '+str(len(new_df_files)) + ' raw files')
    
        dates=[]
        for fname in new_df_files['path']:
            
            datetimestring=re.search('D\d\d\d\d\d\d\d\d-T\d\d\d\d\d\d',fname).group()
            dates.append( pd.to_datetime( datetimestring,format='D%Y%m%d-T%H%M%S' ) )
        new_df_files['date'] = dates
    
    
        new_df_files['to_do']=True 
        
        
        self.df_files=pd.concat([self.df_files,new_df_files])
        self.df_files.drop_duplicates(inplace=True)
        
        self.df_files =  self.df_files.sort_values('date')
        self.df_files=self.df_files.reset_index(drop=True)
        

        # look for already processed data
        self.df_files['to_do']=True    
        
        if os.path.isfile(self.workpath+'/list_of_rawfiles.csv'):
            df_files_done =  pd.read_csv(self.workpath+'/list_of_rawfiles.csv',index_col=0)
            df_files_done=df_files_done.loc[ df_files_done['to_do']==False,: ]
        
            names = self.df_files['path'].apply(lambda x: Path(x).stem)       
            names_done = df_files_done['path'].apply(lambda x: Path(x).stem)  
            # breakpoint()
            
        # print(names)
        # print(nasc_done)
            ix_done= names.isin( names_done  )  
    
        # print(ix_done)
            self.df_files.loc[ix_done,'to_do'] = False        
        self.n_todo=np.sum(self.df_files['to_do'])
        print('To do: ' + str(self.n_todo))
        
        # echogram=pd.DataFrame([])    
        # positions=pd.DataFrame([])    
        
        unit_length_min=pd.to_timedelta(10,'min')
        
        ix_todo = np.where( self.df_files['to_do']==True )[0]
        if self.n_todo>0:
                index = ix_todo[0]
                row = self.df_files.iloc[ index ,:]

        # for index, row in self.df_files.iterrows():
        #     if self.toggle_proc.active & (row['to_do']==True):
                rawfile=row['path']
                print('working on '+rawfile)
                try:
                    
                    # breakpoint()
                    # rawfile =r"C:\Users\a5278\Documents\postdoc_krill\krillscan\source_folder\D20220410-T195719.raw"
                    
                    echogram_file, positions_file , echogram_file_raw = self.read_raw(rawfile)
                    
                    if hasattr(self, 'echogram'):
                        self.echogram = xr.concat([self.echogram,echogram_file], dim="time")
                        self.echogram_raw = xr.concat([self.echogram_raw,echogram_file_raw], dim="time")
                    else:
                        self.echogram=echogram_file
                        self.echogram_raw=echogram_file_raw
                                            
                    # self.echogram = pd.concat([ self.echogram,echogram_file ])
                    
                    
                    self.positions = pd.concat([ self.positions,positions_file ])
                    self.positions=self.positions.reset_index(drop=True)
                    # t=pd.to_datetime( self.echogram.coords['time'].data )
                    t=pd.to_datetime( self.positions['ping_time'].values )
                    
                    # breakpoint()
                    # print(echogram)
                    
                    # print( [ t.max() , t.min() ])
                    
                    while (t.max() - t.min()) > unit_length_min:
                        
                        print( (t.max() - t.min())  )
                        
                        # print(  (t.min() + unit_length_min) > t)
                        ix_end = np.where( (t.min() + unit_length_min) > t )[0][-1]
                        ix_start=t.argmin()
                        # print([ix_start,ix_end])
                        
                        # jump over snipps that are to small
                        if (ix_end-ix_start)<50:
                            # print('echogram to short, jumping over')
                            self.echogram = self.echogram[:,:,ix_end+1:]
                            self.positions = self.positions.iloc[ix_end+1:,:]
                            self.echogram_raw = self.echogram_raw[:,:,ix_end+1:]
                            t=pd.to_datetime( self.positions['ping_time'].values )

                            
                        else:    
                        
                            # accumulate 10 min snippet  
                            new_echogram = self.echogram[:,:,ix_start:ix_end]
                            new_positions = self.positions.iloc[ix_start:ix_end,:]
                            new_echogram_raw = self.echogram_raw[:,:,ix_start:ix_end]

                            self.echogram = self.echogram[:,:,ix_end+1:]
                            self.positions = self.positions.iloc[ix_end+1:,:]
                            self.echogram_raw = self.echogram_raw[:,:,ix_end+1:]

                            t=pd.to_datetime( self.positions['ping_time'].values )
    
                            # try:
                            df_nasc_file, mask_swarm, mask_dbdiff = self.detect_krill_swarms(new_echogram,new_positions)   
                            
                            name = new_positions['ping_time'].min().strftime('D%Y%m%d-T%H%M%S' )    
                            # name = t.min().strftime('D%Y%m%d-T%H%M%S' )         
                            print('saving: '+name)
                            # df_sv_swarm[ new_echogram==-999 ] =-999
                            
                                                    
                            df_nasc_file.to_hdf(self.workpath+'/'+ name + '_nasctable.h5', key='df', mode='w'  )
                            
                            dffloat=df_nasc_file.copy()
                            formats = {'lat': "{:.6f}", 'lon': "{:.6f}", 'distance_m': "{:.4f}",'bottomdepth_m': "{:.1f}",'nasc_swarm': "{:.2f}",'nasc_dbdiff': "{:.2f}"}
                            for col, f in formats.items():
                                dffloat[col] = dffloat[col].map(lambda x: f.format(x))                           
                            # dffloat.to_csv( name + '_nasctable.gzip',compression='gzip' )
                            dffloat.to_csv(self.workpath+'/'+ name + '_nasctable.csv')
                            
                            # df_sv_swarm.astype('float16').to_hdf(self.workpath+'/'+ name + '_sv_swarm.h5', key='df', mode='w'  )
                            
                            # add mask info to xarray
                            t_mask= new_echogram.coords['time']
                            r= new_echogram.coords['depth']
                            
                            df_mask = pd.DataFrame( np.transpose(mask_swarm))
                            df_mask.index=t_mask
                            df_mask.columns=r
                            df_mask.astype('bool').to_hdf(self.workpath+'/'+ name + '_mask_swarm.h5', key='df', mode='w'  )
                               
                            
                            # xx = xr.DataArray(mask_swarm,coords={"time": t,"depth": r,"frequency": 0},  dims=("depth","time"))
                            # new_echogram = xr.concat([new_echogram,xx], dim="frequency")
                            
                            
                            if len(np.shape(mask_dbdiff))>1:
                                df_mask = pd.DataFrame( np.transpose(mask_dbdiff))
                                df_mask.index=t_mask
                                df_mask.columns=r
                                df_mask.astype('bool').to_hdf(self.workpath+'/'+ name + '_mask_dbdiff.h5', key='df', mode='w'  )
                                
                            #    xx = xr.DataArray(mask_dbdiff,coords={"time": t,"depth": r,"frequency": 1},  dims=("depth","time"))
                            #    new_echogram = xr.concat([new_echogram,xx], dim="frequency")
                                                         
                            
                            
                            new_echogram.to_netcdf(self.workpath+'/'+ name + '_echogram.nc')    
                            new_echogram_raw.to_netcdf(self.workpath+'/'+ name + '_rawechogram.nc')    
                            # self.df_files.loc[i,'to_do'] = False
                            # except Exception as e:
                            #   print(e)                      
                    self.df_files.loc[index,'to_do']=False            
                    self.df_files.drop_duplicates(inplace=True)
                    self.df_files=self.df_files.reset_index(drop=True)
                    self.df_files.to_csv(self.workpath+'/list_of_rawfiles.csv')
                   
                except Exception as e:
                    print(e)               
                    print(traceback.format_exc())
                    # breakpoint()
                    self.df_files.loc[index,'to_do']=False            
                    self.df_files.drop_duplicates(inplace=True)
                    self.df_files=self.df_files.reset_index(drop=True)
                    self.df_files.to_csv(self.workpath+'/list_of_rawfiles.csv')                    
                    
                    
        self.callback_process_active==False
                
    def detect_krill_swarms(self,xr_sv,positions):
         # sv= self.echodata[rawfile][ 120000.0] 
         # sv= self.ekdata[ 120000.0]          
         # 
         # breakpoint()
         
         config = configparser.ConfigParser()
         config.read(self.ini_file)            
         f=float(config['GENERAL']['scrutinization_frequency'])
                  
         surface_exclusion_depth_m    = float(self.config['PROCESSING']['surface_exclusion_depth_m'] )
         maximum_depth_m    = float(self.config['PROCESSING']['maximum_depth_m'] )

         
            
         t120 =xr_sv.coords['time'].data
         r120 =xr_sv.coords['depth'].data
         
         # xr_sv.sel(frequency=f).data

         Sv120= xr_sv.sel(frequency=f).data.copy() 
         
         # bottom = self.bottom_detection(Sv120,-38,0)
         # positions['bottom_depth'] = np.take(  r120,bottom.astype(int)  )

         # remove bttom
         
         for i in range(len(t120)):
              ix_na = r120>=  positions['bottomdepth_m'].values[i] 
              Sv120[ix_na,i]=-999

          # plt.figure(0)
          # plt.clf()
          # plt.imshow(Sv120,aspect='auto',vmin=-80)
          # plt.plot( np.arange(len(bottom)), bottom,'-r')
          # plt.colorbar()
          # plt.draw()
          # plt.savefig('t.png')
       
         # breakpoint()
         
         # # get mask for seabed
         # sv2 = Sv120.copy()
         # sv2[np.isnan(sv2)]=-999
         
         # mb = mSB.ariza(sv2, r120, r0=20, r1=1000, roff=0,
         #                  thr=-38, ec=1, ek=(3,3), dc=10, dk=(5,15))
         
         # print('bottom='+str(mb.sum()))
         # Sv120[mb]=-999
        
         ## swarm method

         swarm_sv_threshold_db  = float(self.config['PROCESSING']['swarm_sv_threshold_db'] )

         # get swarms mask
         k = np.ones((3, 3))/3**2
         Sv120cvv = tf.log(convolve2d(tf.lin( Sv120 ), k,'same',boundary='symm'))   
 
         p120           = np.arange(np.shape(Sv120cvv)[1]+1 )                 
         s120           = np.arange(np.shape(Sv120cvv)[0]+1 )           
         m120sh, m120sh_ = mSH.echoview(Sv120cvv, s120, p120, thr=swarm_sv_threshold_db ,
                                    mincan=(3,10), maxlink=(3,15), minsho=(3,15))

         Sv120sw =  Sv120.copy()
         mask_swarm = m120sh.copy()
  
         # Sv120sw[~m120sh] = np.nan  
         # ixdepthvalid= (r120>=20) & (r120<=500)
         # Sv120sw[ ~ixdepthvalid,: ] =np.nan 
         
         Sv120sw[~m120sh] = -999  
         ixdepthvalid= (r120>=surface_exclusion_depth_m) & (r120<=maximum_depth_m )
         Sv120sw[ ~ixdepthvalid,: ] =-999          
         
         cell_thickness=np.abs(np.mean(np.diff( r120) ))               

         r_new = np.arange(0,r120.max(),10)
         # t_new= np.arange(0,Sv120sw.shape[1],10)
        
        
         sv_lin=np.power(10, Sv120sw /10)
        
         sv_downsampled=  resize_local_mean(sv_lin,[ len(r_new) , Sv120sw.shape[1] ] ,grid_mode =True)

         sv_dep=np.transpose(  np.tile(r_new,[sv_downsampled.shape[1],1] ) )
                          
         sa =  integrate.trapezoid(sv_downsampled,sv_dep,axis=0)  
         nasc_swarm =  4*np.pi*1852**2 * sa
         
         nasc_swarm_rs =nasc_swarm


         df_sv_swarm=pd.DataFrame( np.transpose(Sv120) )
         df_sv_swarm.index=t120
         df_sv_swarm.columns=r120
          # print('df_sv')
         
         df_nasc_file=pd.DataFrame([])
         # df_nasc_file['time']=positions['ping_time']
         df_nasc_file['lat']=positions['latitude']
         df_nasc_file['lon']=positions['longitude']
         df_nasc_file['distance_m']=np.append(np.array([0]),geod.line_lengths(lons=positions['longitude'],lats=positions['latitude']) )
         df_nasc_file['bottomdepth_m']=positions['bottomdepth_m']
         
         # # breakpoint()
         # bottomdepth=[]         
         # for j in range(Sv120.shape[1]):
         #     row_1=Sv120[:,j]
         #     if np.sum(row_1==-999)>0:
         #         bottomdepth.append( np.min(r120[row_1==-999]) )
         #     else:
         #         bottomdepth.append( r120.max() )
         # # print(bottomdepth)   
         # df_nasc_file['bottomdepth_m']=bottomdepth
            
           
         df_nasc_file['nasc_swarm']=nasc_swarm_rs
         df_nasc_file.index=positions['ping_time']
         
         # df_nasc_file=df_nasc_file.resample('5s').mean()
         print('Krill detection complete: '+str(np.mean(nasc_swarm)) ) 
         
         
         ## db difference method
         dbdiff_sv_threshold_db   = float(self.config['PROCESSING']['dbdiff_sv_threshold_db'] )

         if np.sum( np.isin( [38000.0, 120000.0],xr_sv.coords['frequency'].data ) ) ==2 :
         
             Sv38= xr_sv.sel(frequency=38000.0).data 
             # remove bttom
             for i in range(len(t120)):
                 ix_na = r120>=  positions['bottomdepth_m'].values[i]-10 
                 Sv38[ix_na,i]=np.nan 
             
             db_diff= Sv120 -Sv38
             
             mask_dbdiff = db_diff>dbdiff_sv_threshold_db 
             
             Sv120db=Sv120.copy()
             Sv120db[~mask_dbdiff]=np.nan
             
             ixdepthvalid= (r120>=surface_exclusion_depth_m) & (r120<=maximum_depth_m )
             Sv120db[~ixdepthvalid,:]=np.nan
             
             cell_thickness=np.abs(np.mean(np.diff( r120) ))               
             nasc_dbdiff=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120db /10)*cell_thickness ,axis=0)   
             df_nasc_file['nasc_dbdiff']=nasc_dbdiff
         else:
             df_nasc_file['nasc_dbdiff']=np.nan
             mask_dbdiff=np.nan
             
         return df_nasc_file, mask_swarm, mask_dbdiff
 
        
         
    # def start():
    #     print('start')

    def callback_email(self):
      if  (self.callback_email_active==False) :      
        self.callback_email_active==True
        print('checking wether to send email')
        self.config = configparser.ConfigParser()
        self.config.read(self.ini_file)   
                
        emailfrom = self.config['EMAIL']['email_from']
        emailto = self.config['EMAIL']['email_to']
        password = str(self.config['EMAIL']['pw'])
        # fileToSend = r"D20220212-T180420_nasctable.h5"
        # username = "raw2nasc"
        # password = "raw2nasckrill"
        # print(self.config['EMAIL']['email_send'])
        email_send =int(self.config['EMAIL']['email_send'])
        # print(email_send)
        if email_send>0:
            # breakpoint()
            
            # self.workpath=  os.path.join(self.folder_source,'krill_data')
            
            # os.chdir(self.workpath)
            # self.df_files=pd.read_csv(self.workpath+'/list_of_rawfiles.csv')
           
            nasc_done =  pd.DataFrame( glob.glob( self.workpath+'/*_nasctable.h5' ) )
            if len(nasc_done)>0:               
                if os.path.isfile(self.workpath+'/list_of_sent_files.csv'):
                    df_files_sent =  pd.read_csv(self.workpath+'/list_of_sent_files.csv',index_col=0)
                    ix_done= nasc_done.iloc[:,0].isin( df_files_sent.iloc[:,0]  )  
                    nasc_done=nasc_done[~ix_done]
                
                else:    
                    df_files_sent=pd.DataFrame([])
                
                nascfile_times=[]
                for fname in nasc_done.iloc[:,0]:         
                    datetimestring=re.search('D\d\d\d\d\d\d\d\d-T\d\d\d\d\d\d',fname).group()
                    nascfile_times.append( pd.to_datetime( datetimestring,format='D%Y%m%d-T%H%M%S' ) )
                
                # nascfile_times=pd.to_datetime( nasc_done.iloc[:,0] ,format='D%Y%m%d-T%H%M%S_nasctable.h5' )
                nasc_done=nasc_done.iloc[np.argsort(nascfile_times),0].values
                     
                n_files=int(self.config['EMAIL']['files_per_email'])
                send_echograms=int(self.config['EMAIL']['send_echograms'])
                echogram_resolution_in_seconds=str(self.config['EMAIL']['echogram_resolution_in_seconds'])
                print( str(len(nasc_done)) +' files that can be sent')
    
                while (len(nasc_done)>n_files) :
                    
                    
                    files_to_send=nasc_done[0:n_files]
                    # print(nasc_done)
                    
                    msg = MIMEMultipart()
                    msg["From"] = emailfrom
                    msg["To"] = emailto
                    msg["Subject"] = "Krillscan data from "+ self.config['GENERAL']['vessel_name']+' ' +files_to_send[0][-30:-13]+'_to_'+files_to_send[-1][-30:-13]
                  
                    msgtext = str(dict(self.config['GENERAL']))
                    msg.attach(MIMEText( msgtext   ,'plain'))
    
                    loczip = msg["Subject"]+'.zip'
                    zip = zipfile.ZipFile(loczip, "w", zipfile.ZIP_DEFLATED)
                    zip.write(self.ini_file)
    
                    for fi in files_to_send:   
                        zip.write(fi,arcname=fi[-30:]  )                                  
    
                    if send_echograms>0:                       
                        for fi in files_to_send:      
                            
                            
                            # breakpoint()

                            xr_sv = xr.open_dataarray(self.workpath+'/'+fi[-30:-13] + '_echogram.nc')
                            
                            f=float(self.config['GENERAL']['scrutinization_frequency'])

              
                            ix_f = np.where( xr_sv.coords['frequency'].values==f)[0][0] 
                            
                            sv = np.transpose( np.squeeze( xr_sv[ix_f,:,:].data) )

                            df=pd.DataFrame(sv)
                            df.index= xr_sv.coords['time'].values
                            df.columns= xr_sv.coords['depth'].values
                                                        
                            
                            df=df.resample(echogram_resolution_in_seconds+'s').mean()
                            targetname=fi[-30:-13] + '_sv_swarm_mail.h5' 
                            df.astype('float16').to_hdf(targetname,key='df',mode='w')
                            # df.astype('float16').to_csv(targetname,compression='gzip')
                            zip.write(targetname)                                                      
                            os.remove(targetname)
                    
                    zip.close()
                    fp = open(loczip, "rb")
                    attachment = MIMEBase('application', 'x-zip')
                    attachment.set_payload(fp.read())
                    fp.close()
                    encoders.encode_base64(attachment)
                    attachment.add_header("Content-Disposition", "attachment", filename=loczip)
                    msg.attach(attachment)    
                    
                    os.remove(loczip)
    
                    try:        
                        ctx = ssl.create_default_context()
                        server = smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=ctx)
                        
                        server.login(emailfrom, password)
                        
                        # print(df_files_sent)
                    
                        server.sendmail(emailfrom, emailto.split(','), msg.as_string())
                        if len(df_files_sent)>0:
                            df_files_sent= pd.concat([pd.Series(df_files_sent.iloc[:,0].values),pd.DataFrame(files_to_send)],ignore_index=True)
                        else:
                            df_files_sent=pd.DataFrame(files_to_send)
                            
                        # df_files_sent=df_files_sent.reset_index(drop=True)
                        df_files_sent=df_files_sent.drop_duplicates()
                        df_files_sent.to_csv(self.workpath+'/list_of_sent_files.csv')
                        
                        
                        print('email sent: ' +   msg["Subject"] )
                        nasc_done=nasc_done[n_files::]
                        server.quit()
    
                    except Exception as e:
                        print(e)
                                        
        
        self.callback_email_active==False
        
#%%

ks=krillscan_class()
# ks.start('settings.ini')


# ks.stop()