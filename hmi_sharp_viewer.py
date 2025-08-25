"""
HMI SHARP Data Viewer - Interactive Streamlit App with Plotly
Author: Solar Physics Educational Tool
Description: Download and visualize HMI SHARP magnetogram data interactively
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
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

st.markdown("""
    <style>
    .main .block-container {
        max-width: 1440px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

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

def create_interactive_plot(data1, data2, title1, title2, colorscale1='RdBu', colorscale2='gray',
                          zmin1=None, zmax1=None, zmin2=None, zmax2=None,
                          colorbar_title1="", colorbar_title2=""):
    """Create interactive subplot with Plotly"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(title1, title2),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # First subplot
    fig.add_trace(
        go.Heatmap(
            z=data1,
            colorscale=colorscale1,
            zmin=zmin1,
            zmax=zmax1,
            colorbar=dict(
                title=colorbar_title1,
                x=1.02,
                y=0.75,
                len=0.4,
                thickness=15
            ),
            hovertemplate=f"{colorbar_title1}: %{{z:.1f}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Second subplot
    fig.add_trace(
        go.Heatmap(
            z=data2,
            colorscale=colorscale2,
            zmin=zmin2,
            zmax=zmax2,
            colorbar=dict(
                title=colorbar_title2,
                x=1.02,
                y=0.25,
                len=0.4,
                thickness=15
            ),
            hovertemplate=f"{colorbar_title2}: %{{z:.1f}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(title_text="X [pixels]", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Y [pixels]", row=1, col=1)
    fig.update_yaxes(title_text="Y [pixels]", row=2, col=1)

    # ADD these lines to link the axes:
    fig.update_xaxes(matches='x2', row=1, col=1)  # Link top x-axis to bottom
    fig.update_yaxes(matches='y2', row=1, col=1)  # Link top y-axis to bottom
    
    # Make plots square with equal aspect ratio
    fig.update_xaxes(constrain="domain", row=1, col=1)
    fig.update_xaxes(constrain="domain", row=2, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=2, col=1)
    
    return fig

def create_single_plot(data, title, colorscale='RdBu', zmin=None, zmax=None, colorbar_title=""):
    """Create single interactive plot with Plotly"""
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title=colorbar_title,
            thickness=20
        ),
        hovertemplate=f"{colorbar_title}: %{{z:.1f}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        height=600,
        xaxis_title="X [pixels]",
        yaxis_title="Y [pixels]",
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig

# Main app
def main():
    st.title("☀️ HMI SHARP Data Viewer")
    st.markdown("""
    This interactive tool allows you to download and visualize HMI SHARP magnetogram data 
    from the Solar Dynamics Observatory (SDO). The plots are fully interactive - you can zoom, 
    pan, and hover to see exact values!
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
                min_value=datetime(2010, 5, 1),  # SDO/HMI started in 2010
                max_value=datetime.now(),  # Allow up to current date
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
        
        plot_type = st.radio(
            "Select Plot Type",
            ["Magnetic Field (Bz)", "Field Inclination", "All Components", "Field Strength"],
            help="Choose which magnetic field property to visualize"
        )
        
        colormap_bz = st.selectbox(
            "Magnetic Field Colormap",
            ["RdBu", "seismic", "bwr", "coolwarm", "RdBu_r"],
            index=4,
            help="Select colormap for magnetic field visualization"
        )
        
        show_contours = st.checkbox("Show Contours", value=False,
                                   help="Overlay contour lines on the plots")
        
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
            st.header("📈 Interactive Visualizations")
            st.info("💡 **Tip:** Use the tools in the top-right of each plot to zoom, pan, reset view, or download the image!")
            
            # Create plots based on selection
            if plot_type == "Magnetic Field (Bz)":
                st.subheader("Vertical Magnetic Field (Bz) and Continuum Intensity")
                
                fig = create_interactive_plot(
                    bz_new, ic_new,
                    "Vertical Magnetic Field (Bz)", "Continuum Intensity",
                    colorscale1=colormap_bz, colorscale2='gray',
                    zmin1=-1500, zmax1=1500,
                    colorbar_title1="Bz [G]", colorbar_title2="Intensity"
                )
                
                if show_contours:
                    # Add contours to Bz
                    fig.add_trace(
                        go.Contour(
                            z=bz_new,
                            showscale=False,
                            contours=dict(
                                start=-1500,
                                end=1500,
                                size=300,
                                coloring='lines'
                            ),
                            line=dict(width=1),
                            opacity=0.3
                        ),
                        row=1, col=1
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "Field Inclination":
                st.subheader("Magnetic Field Inclination and Continuum Intensity")
                
                # Calculate inclination
                theta_inc_new = np.rad2deg(np.arctan2(bz_new, 
                                          np.sqrt(bx_new**2 + by_new**2)))
                
                fig = create_interactive_plot(
                    theta_inc_new, ic_new,
                    "Field Inclination", "Continuum Intensity",
                    colorscale1='RdYlGn', colorscale2='gray',
                    zmin1=-90, zmax1=90,
                    colorbar_title1="Angle [°]", colorbar_title2="Intensity"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "All Components":
                st.subheader("All Magnetic Field Components")
                
                # Create 2x2 subplot
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Bx (East-West)", "By (North-South)", 
                                  "Bz (Vertical)", "Total Field Strength"),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.12
                )
                
                # Calculate total field
                b_total = np.sqrt(bx_new**2 + by_new**2 + bz_new**2)
                
                # Add all components
                components = [
                    (bx_new, "Bx [G]", 1, 1),
                    (by_new, "By [G]", 1, 2),
                    (bz_new, "Bz [G]", 2, 1),
                    (b_total, "|B| [G]", 2, 2)
                ]
                
                for i, (data, label, row, col) in enumerate(components):
                    if row == 2 and col == 2:  # Total field
                        colorscale = 'Viridis'
                        zmin, zmax = 0, np.nanmax(b_total)
                    else:
                        colorscale = colormap_bz
                        vmax = np.nanmax(np.abs(data))
                        zmin, zmax = -vmax, vmax
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=data,
                            colorscale=colorscale,
                            zmin=zmin,
                            zmax=zmax,
                            colorbar=dict(
                                title=label,
                                x=1.02 if col == 2 else -0.15,
                                y=0.75 if row == 1 else 0.25,
                                len=0.35,
                                thickness=12
                            ),
                            hovertemplate=f"{label}: %{{z:.1f}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>"
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(height=800, showlegend=False)

                # Link all axes together
                fig.update_xaxes(matches='x', row=1, col=2)  # Link all x-axes to x1
                fig.update_xaxes(matches='x', row=2, col=1)
                fig.update_xaxes(matches='x', row=2, col=2)

                fig.update_yaxes(matches='y', row=1, col=2)  # Link all y-axes to y1
                fig.update_yaxes(matches='y', row=2, col=1)
                fig.update_yaxes(matches='y', row=2, col=2)

                fig.update_xaxes(title_text="X [pixels]", row=2)
                fig.update_yaxes(title_text="Y [pixels]", col=1)
                
                # Fix aspect ratio for all subplots
                for row in [1, 2]:
                    for col in [1, 2]:
                        fig.update_xaxes(constrain="domain", row=row, col=col)
                        fig.update_yaxes(scaleanchor=f"x{col + (row-1)*2}", scaleratio=1, row=row, col=col)
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Field Strength
                st.subheader("Total Magnetic Field Strength")
                
                b_total = np.sqrt(bx_new**2 + by_new**2 + bz_new**2)

                fig = create_interactive_plot(
                    b_total, ic_new,
                    "Total Field Strength |B|", "Continuum Intensity",
                    colorscale1='Viridis', colorscale2='gray',
                    zmin1=0, zmax1=np.nanmax(b_total),
                    colorbar_title1="Angle [°]", colorbar_title2="Intensity"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.header("📊 Field Statistics")
            
            # Calculate statistics
            b_total = np.sqrt(bx_new**2 + by_new**2 + bz_new**2)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Max Bz", f"{np.nanmax(bz_new):.1f} G")
                st.metric("Max Bx", f"{np.nanmax(np.abs(bx_new)):.1f} G")
            with col2:
                st.metric("Min Bz", f"{np.nanmin(bz_new):.1f} G")
                st.metric("Max By", f"{np.nanmax(np.abs(by_new)):.1f} G")
            with col3:
                st.metric("Mean |B|", f"{np.nanmean(b_total):.1f} G")
                st.metric("Std |B|", f"{np.nanstd(b_total):.1f} G")
            with col4:
                st.metric("Max |B|", f"{np.nanmax(b_total):.1f} G")
                st.metric("Flux", f"{np.sum(np.abs(bz_new))*0.36*0.36:.2e} Mx")
            
            # Additional analysis
            with st.expander("📊 Advanced Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Magnetic Field Distribution**")
                    # Create histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=bz_new.flatten(),
                        nbinsx=50,
                        name="Bz Distribution",
                        marker_color='blue',
                        opacity=0.7
                    ))
                    fig_hist.update_layout(
                        title="Vertical Field Distribution",
                        xaxis_title="Bz [G]",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    st.write("**Polarity Analysis**")
                    positive_flux = np.sum(bz_new[bz_new > 0]) * 0.36 * 0.36
                    negative_flux = np.sum(np.abs(bz_new[bz_new < 0])) * 0.36 * 0.36
                    
                    st.metric("Positive Flux", f"{positive_flux:.2e} Mx")
                    st.metric("Negative Flux", f"{negative_flux:.2e} Mx")
                    st.metric("Net Flux", f"{(positive_flux - negative_flux):.2e} Mx")
                    st.metric("Flux Imbalance", f"{abs(positive_flux - negative_flux)/(positive_flux + negative_flux)*100:.1f}%")
            
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
        5. **Choose visualization options** in the sidebar
        6. **Click "Download and Visualize Data"** to fetch and display the magnetogram
        
        ### 🔍 Interactive Features:
        - **Zoom**: Click and drag to zoom into a specific region
        - **Pan**: Hold shift and drag to pan around
        - **Hover**: Move your mouse over the plot to see exact values
        - **Reset**: Double-click to reset the view
        - **Download**: Use the camera icon to save the plot
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
        
        # Visual guide
        st.subheader("🎨 Understanding the Visualizations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("""
            **🔴🔵 Vertical Field (Bz)**
            - Red: Positive polarity (North)
            - Blue: Negative polarity (South)
            - Intensity: Field strength
            """)
        
        with col2:
            st.write("""
            **🟢🟡🔴 Field Inclination**
            - Green: Horizontal field
            - Yellow: Inclined field
            - Red: Vertical field
            """)
        
        with col3:
            st.write("""
            **🟣 Total Field Strength**
            - Purple/Yellow: Strong field
            - Dark: Weak field
            - Shows overall magnetic energy
            """)

if __name__ == "__main__":
    main()