"""
HMI SHARP Data Viewer - Interactive Streamlit App
Author: Solar Physics Educational Tool
Description: Download and visualize HMI SHARP magnetogram data interactively
"""

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import sunpy
import sunpy.map
from sunpy.time import parse_time
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
import pandas as pd
import drms
import os
from pathlib import Path
from glob import glob
import urllib.request
import warnings
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="HMI SHARP Data Viewer",
    page_icon="☀️",
    layout="wide"
)

# Helper function for colorbar
def plot_colorbar(im, ax, bbox_to_anchor=(1.02, 0., 0.1, 1), fontsize=10,
                  orientation="vertical", title=None, scilimits=(-4, 4), **kwargs):
    clb_ax = ax.inset_axes(bbox_to_anchor, transform=ax.transAxes)
    clb = plt.colorbar(im, pad=0.05, orientation=orientation, ax=ax, cax=clb_ax, **kwargs)
    clb_ax.tick_params(labelsize=fontsize)
    
    if orientation == "vertical":
        clb_ax.ticklabel_format(axis="y", style="sci", scilimits=scilimits)
        clb_ax.yaxis.get_offset_text().set_fontsize(fontsize)
        clb_ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    elif orientation == "horizontal":
        clb_ax.ticklabel_format(axis="x", style="sci", scilimits=scilimits)
        clb_ax.xaxis.get_offset_text().set_fontsize(fontsize)
        clb_ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    if title is not None:
        clb.set_label(title, fontsize=fontsize)
    
    return clb, clb_ax

# Download function
@st.cache_data
def download_sharp(noaa_num, dateobs, email_address, overwrite=False,
                   harp_noaa_list="./downloads/all_harps_with_noaa_ars.txt"):
    """Download SHARP data for given NOAA number and date"""
    
    out_dir = Path("./downloads")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    
    # Download the NOAA to HARP conversion table
    if not os.path.exists(harp_noaa_list):
        url = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"
        urllib.request.urlretrieve(url, harp_noaa_list)
    
    # Read conversion table
    harp_noaa_df = pd.read_csv(harp_noaa_list, sep=r"\s+", header=None, skiprows=1,
                               names=["harp_num", "noaa_num"])
    
    # Find HARP number
    matching_harps = harp_noaa_df.loc[harp_noaa_df["noaa_num"].str.contains(str(noaa_num)), "harp_num"]
    if len(matching_harps) == 0:
        st.error(f"No HARP number found for NOAA {noaa_num}")
        return None
    
    harp_num = matching_harps.values[0]
    
    # Check for existing files
    if not overwrite:
        exist_files = glob(os.path.join(out_dir, f"hmi.sharp_cea_720s.[{harp_num:.0f}]*"))
        
        if len(exist_files) > 0:
            exist_times_str = [fname.split(".")[3][:-4] for fname in exist_files]
            exist_times = parse_time(exist_times_str, scale="tai")
            time_diff = np.abs(exist_times - Time(dateobs))
            time_diff = np.array([diff.to_value(u.min) for diff in time_diff])
            
            if np.any(time_diff < 6):
                st.info("Using existing data files (found similar timestamps).")
                return harp_num
    
    # Export and download data
    client = drms.Client()
    qstr = f"hmi.sharp_cea_720s[{harp_num:.0f}][{dateobs}]"
    
    with st.spinner(f"Exporting data for HARP {harp_num:.0f}..."):
        result = client.export(qstr, method="url", email=email_address, protocol="fits")
        result.wait()
        
        if result.has_succeeded():
            st.success("Export successful! Downloading files...")
            result.download(out_dir)
        else:
            st.error("Export failed. Please check your inputs and try again.")
            return None
    
    return harp_num

# Load data function
@st.cache_data
def load_sharp_data(harp_num):
    """Load SHARP FITS files and return data arrays"""
    
    # Get filenames
    bp_files = glob(f"./downloads/hmi.sharp_cea_720s.[{harp_num:.0f}]*Bp.fits")
    bt_files = glob(f"./downloads/hmi.sharp_cea_720s.[{harp_num:.0f}]*Bt.fits")
    br_files = glob(f"./downloads/hmi.sharp_cea_720s.[{harp_num:.0f}]*Br.fits")
    ic_files = glob(f"./downloads/hmi.sharp_cea_720s.[{harp_num:.0f}]*continuum.fits")
    
    if not (bp_files and bt_files and br_files and ic_files):
        st.error("Some required FITS files are missing!")
        return None, None, None, None
    
    # Load the most recent files
    bp_file = sorted(bp_files)[-1]
    bt_file = sorted(bt_files)[-1]
    br_file = sorted(br_files)[-1]
    ic_file = sorted(ic_files)[-1]
    
    # Load data
    with fits.open(bp_file) as hdul:
        bx_new = hdul[1].data[:, :]
    
    with fits.open(bt_file) as hdul:
        by_new = -hdul[1].data[:, :]  # SHARP bt is southward
    
    with fits.open(br_file) as hdul:
        bz_new = hdul[1].data[:, :]
    
    with fits.open(ic_file) as hdul:
        ic_new = hdul[1].data[:, :]
    
    return bx_new, by_new, bz_new, ic_new

# Main app
def main():
    st.title("☀️ HMI SHARP Data Viewer")
    st.markdown("""
    This interactive tool allows you to download and visualize HMI SHARP magnetogram data 
    from the Solar Dynamics Observatory (SDO). Simply enter a NOAA Active Region number 
    and observation date to get started.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Data Selection")
        
        # Email registration notice
        st.info("""
        **Important:** Before using this app, please register your email at:
        [JSOC Email Registration](http://jsoc.stanford.edu/ajax/register_email.html)
        """)
        
        # Input fields
        col1, col2 = st.columns(2)
        with col1:
            noaa_number = st.number_input(
                "NOAA AR Number",
                min_value=10000,
                max_value=99999,
                value=12017,
                help="Enter the NOAA Active Region number (e.g., 12017)"
            )
        
        with col2:
            date_input = st.date_input(
                "Observation Date",
                value=datetime(2014, 3, 29),
                help="Select the observation date"
            )
        
        time_input = st.time_input(
            "Observation Time (UTC)",
            value=datetime.strptime("17:36:00", "%H:%M:%S").time(),
            help="Select the observation time in UTC"
        )
        
        # Combine date and time
        dateobs = f"{date_input}T{time_input}"
        
        email = st.text_input(
            "Email Address",
            value="",
            placeholder="your.email@example.com",
            help="Enter your registered JSOC email address"
        )
        
        # Visualization options
        st.header("🎨 Visualization Options")
        
        show_bz = st.checkbox("Show Vertical Magnetic Field (Bz)", value=True)
        show_inclination = st.checkbox("Show Field Inclination", value=True)
        
        # Zoom options
        use_zoom = st.checkbox("Enable Zoom", value=False)
        if use_zoom:
            col1, col2 = st.columns(2)
            with col1:
                x_min = st.number_input("X min", value=350, step=10)
                x_max = st.number_input("X max", value=550, step=10)
            with col2:
                y_min = st.number_input("Y min", value=200, step=10)
                y_max = st.number_input("Y max", value=400, step=10)
        
        # Download button
        download_button = st.button("🚀 Download and Visualize Data", type="primary")
    
    # Main content area
    if download_button:
        if not email:
            st.error("Please enter your registered JSOC email address!")
            return
        
        # Download data
        st.header("📥 Downloading Data")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Searching for HARP number...")
            progress_bar.progress(20)
            
            harp_num = download_sharp(noaa_number, dateobs, email_address=email)
            
            if harp_num is None:
                return
            
            status_text.text(f"Found HARP {harp_num:.0f}. Loading data...")
            progress_bar.progress(60)
            
            # Load data
            bx_new, by_new, bz_new, ic_new = load_sharp_data(harp_num)
            
            if bz_new is None:
                return
            
            progress_bar.progress(100)
            status_text.text("Data loaded successfully!")
            
            # Display information
            st.success(f"✅ Successfully loaded data for NOAA {noaa_number} (HARP {harp_num:.0f})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NOAA Number", noaa_number)
            with col2:
                st.metric("HARP Number", f"{harp_num:.0f}")
            with col3:
                st.metric("Data Shape", f"{bz_new.shape[0]} × {bz_new.shape[1]}")
            
            # Create visualizations
            st.header("📈 Visualizations")
            
            # Vertical magnetic field plot
            if show_bz:
                st.subheader("Vertical Magnetic Field (Bz) and Continuum Intensity")
                
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), 
                                                layout="constrained",
                                                sharex=True, sharey=True)
                
                im1 = ax1.imshow(bz_new, origin="lower", interpolation="none",
                                vmin=-1500, vmax=1500, cmap="RdBu_r")
                plot_colorbar(im1, ax1, bbox_to_anchor=(1.02, 0, 0.03, 1))
                ax1.set_title("Bz [Gauss]", fontsize=14)
                ax1.set_ylabel("Y [pixels]")
                
                im2 = ax2.imshow(ic_new, origin="lower", interpolation="none",
                                cmap="gray")
                ax2.set_title("Continuum Intensity", fontsize=14)
                ax2.set_xlabel("X [pixels]")
                ax2.set_ylabel("Y [pixels]")
                
                for ax_ in (ax1, ax2):
                    ax_.grid(True, lw=0.8, alpha=0.8, color="w", ls=":")
                    ax_.set_aspect(1)
                    if use_zoom:
                        ax_.set_xlim(x_min, x_max)
                        ax_.set_ylim(y_min, y_max)
                
                st.pyplot(fig1)
            
            # Field inclination plot
            if show_inclination:
                st.subheader("Magnetic Field Inclination and Continuum Intensity")
                
                # Calculate inclination
                theta_inc_new = np.rad2deg(np.arctan2(bz_new, 
                                          np.sqrt(bx_new**2 + by_new**2)))
                
                fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12),
                                                layout="constrained",
                                                sharex=True, sharey=True)
                
                im1 = ax1.imshow(theta_inc_new, origin="lower", interpolation="none",
                                vmin=-90, vmax=90, cmap="RdYlGn")
                plot_colorbar(im1, ax1, bbox_to_anchor=(1.02, 0, 0.03, 1))
                ax1.set_title("Inclination [Degrees]", fontsize=14)
                ax1.set_ylabel("Y [pixels]")
                
                im2 = ax2.imshow(ic_new, origin="lower", interpolation="none",
                                cmap="gray")
                ax2.set_title("Continuum Intensity", fontsize=14)
                ax2.set_xlabel("X [pixels]")
                ax2.set_ylabel("Y [pixels]")
                
                for ax_ in (ax1, ax2):
                    ax_.grid(True, lw=0.8, alpha=0.8, color="w", ls=":")
                    ax_.set_aspect(1)
                    if use_zoom:
                        ax_.set_xlim(x_min, x_max)
                        ax_.set_ylim(y_min, y_max)
                
                st.pyplot(fig2)
            
            # Statistics
            st.header("📊 Field Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Max Bz", f"{np.nanmax(bz_new):.1f} G")
            with col2:
                st.metric("Min Bz", f"{np.nanmin(bz_new):.1f} G")
            with col3:
                st.metric("Mean |B|", f"{np.nanmean(np.sqrt(bx_new**2 + by_new**2 + bz_new**2)):.1f} G")
            with col4:
                st.metric("Max |B|", f"{np.nanmax(np.sqrt(bx_new**2 + by_new**2 + bz_new**2)):.1f} G")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your inputs and ensure you have registered your email at JSOC.")
    
    else:
        # Instructions when no data is loaded
        st.info("""
        ### 📖 How to use this app:
        
        1. **Register your email** at [JSOC](http://jsoc.stanford.edu/ajax/register_email.html) if you haven't already
        2. **Enter the NOAA AR number** of the region you want to study
        3. **Select the observation date and time** (UTC)
        4. **Enter your registered email address**
        5. **Click "Download and Visualize Data"** to fetch and display the magnetogram
        
        The app will automatically:
        - Convert NOAA number to HARP number
        - Download the SHARP data from JSOC
        - Display vertical magnetic field and inclination maps
        - Show continuum intensity for context
        """)
        
        # Example regions
        st.subheader("📚 Example Active Regions")
        example_data = {
            "NOAA": [12017, 12192, 12673, 13664],
            "Date": ["2014-03-29", "2014-10-24", "2017-09-06", "2024-05-10"],
            "Description": [
                "X-class flare producer",
                "Largest sunspot group of Solar Cycle 24",
                "Produced X9.3 flare",
                "Recent major flare producer"
            ]
        }
        st.table(pd.DataFrame(example_data))

if __name__ == "__main__":
    main()