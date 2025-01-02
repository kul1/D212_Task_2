from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import os
from utils.config_loader import load_config
from datetime import date

# Step 1: Load the configuration
config = load_config()
print(f"Loaded configuration for {config.CONFIG_TYPE}.")

# Initialize the document
document = Document()

# Utility functions
def add_code_snippet(document, code_text):
    """Add a formatted code snippet to the document."""
    p = document.add_paragraph()

    # Set background shading
    shading_elm = OxmlElement("w:shd")
    shading_elm.set(qn("w:val"), "clear")
    shading_elm.set(qn("w:fill"), "FFFFCC")  # Light yellow background

    pPr = p._element.get_or_add_pPr()
    pPr.append(shading_elm)

    # Add border
    borders = OxmlElement("w:pBdr")
    for side in ["top", "left", "bottom", "right"]:
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "6")  # Border thickness
        border.set(qn("w:space"), "4")
        border.set(qn("w:color"), "auto")
        borders.append(border)
    pPr.append(borders)

    # Add text with Courier font
    r = p.add_run(code_text)
    r.font.name = "Courier New"
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0, 0, 0)
    p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

def add_visual(document, image_path, caption):
    """Add a visual with caption."""
    if os.path.exists(image_path):
        document.add_picture(image_path, width=Inches(5.5))
        p = document.add_paragraph(caption)
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    else:
        p = document.add_paragraph(f"Visual not found: {caption}")
        p.italic = True

def add_apa_references(document, references):
    """Add APA-style references."""
    document.add_heading("References", level=1)
    for ref in references:
        citation = f"{ref['author']} ({ref['year']}). {ref['title']}. Retrieved from {ref['source']}"
        document.add_paragraph(citation, style="Normal")

# Main report function
def create_report():
    document = Document()

    # Title page
    document.add_heading("Data Mining II — D212", level=0).alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    document.add_paragraph("Task 2: Principal Component Analysis (PCA)", style='Title').alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    document.add_paragraph("by").alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    document.add_paragraph("Prateep Kul").bold = True
    document.add_paragraph(f"{date.today().strftime('%B %d, %Y')}", style="Normal").alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    document.add_page_break()

    # Table of Contents
    document.add_heading("Table of Contents", level=1)
    toc = [
        "1. Part I: Research Question",
        "2. Part II: Method Justification",
        "3. Part III: Data Preparation",
        "4. Part IV: Analysis",
        "5. References"
    ]
    for item in toc:
        document.add_paragraph(item, style="List Number")
    document.add_page_break()

    # Part I
    document.add_heading("Part I: Research Question", level=1)
    document.add_paragraph(
        "The research question focuses on understanding the factors contributing to overweight patients, "
        "significantly impacting cost-effective treatment plans through PCA."
    )
    document.add_page_break()

    # Part II
    document.add_heading("Part II: Method Justification", level=1)
    document.add_paragraph(
        "Principal Component Analysis (PCA) reduces dimensionality by identifying components explaining variance. "
        "The technique uncovers relationships between variables and improves interpretability."
    )
    document.add_page_break()

    # Part III
    document.add_heading("Part III: Data Preparation", level=1)

    # Section 1
    document.add_heading("1. Identifying Continuous Data Variables", level=2)
    document.add_paragraph(
        "All numeric columns (except the target column) were dynamically selected for PCA."
    )
    add_code_snippet(document, """# Select continuous columns for PCA
feature_columns = [col for col in data.columns if col != config.TARGET_COLUMN]
numeric_columns = data[feature_columns].select_dtypes(include=['number']).columns

X = data[numeric_columns]
y = data[config.TARGET_COLUMN]
""")

    # Section 2
    document.add_heading("2. Standardizing Continuous Data Variables", level=2)
    document.add_paragraph(
        "Standardization ensures variables contribute equally to PCA by scaling them to mean 0 and standard deviation 1."
    )
    add_code_snippet(document, """from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
""")

    # Section 3
    document.add_heading("3. Performing PCA", level=2)
    document.add_paragraph(
        "PCA was performed using the specified number of components, transforming the data to its principal components."
    )
    add_code_snippet(document, """from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=config.PCA_COMPONENTS_RETAINED)
X_pca = pca.fit_transform(X_scaled)
""")

    # Visualizations
    document.add_heading("Visualizations", level=2)
    document.add_paragraph(
        "The following visualizations illustrate the distribution of features before and after standardization, "
        "as well as the explained variance by PCA components."
    )

    # Visualizations with paths from `config.VISUALS_DIR`
    add_visual(document, os.path.join(config.VISUALS_DIR, "original_distribution.png"),
               "Figure 2: Distribution of Features (Before Standardization)")
    add_visual(document, os.path.join(config.VISUALS_DIR, "standardized_distribution.png"),
               "Figure 3: Distribution of Features (After Standardization)")
    add_visual(document, os.path.join(config.VISUALS_DIR, "explained_variance.png"),
               "Figure 4: Explained Variance by PCA Components")

    print("Part III: Data Preparation added to report.")

    # Part IV
    document.add_heading("Part IV: Analysis", level=1)
    add_visual(document, os.path.join(config.VISUALS_DIR, "correlation_heatmap.png"),
               "Figure 5: Correlation Heatmap")
    document.add_paragraph(
        "The analysis demonstrates how principal components capture maximum variance, revealing important patterns."
    )

    # References
    add_apa_references(document, [
        {"author": "Pedregosa, F. et al.", "year": "2011", "title": "Scikit-learn: Machine Learning in Python",
         "source": "https://jmlr.org/papers/v12/pedregosa11a.html"},
        {"author": "Hunter, J. D.", "year": "2007", "title": "Matplotlib: A 2D Graphics Environment",
         "source": "https://doi.org/10.1109/MCSE.2007.55"},
        {"author": "McKinney, W.", "year": "2010", "title": "Data Structures for Statistical Computing in Python",
         "source": "https://conference.scipy.org/proceedings/scipy2010/pdfs/mckinney.pdf"},
        {"author": "Waskom, M. L.", "year": "2017", "title": "Seaborn: Statistical Data Visualization",
         "source": "https://seaborn.pydata.org/"},
        {"author": "Oliphant, T. E.", "year": "2006", "title": "A Guide to NumPy",
         "source": "https://numpy.org/"}
    ])

    # Save the document
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(config.RESULTS_DIR, "PCA_Report.docx")
    document.save(output_path)
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    create_report()
