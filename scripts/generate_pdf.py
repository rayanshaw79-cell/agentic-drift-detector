from markdown_pdf import MarkdownPdf, Section

md_path = r"c:\Users\acer\agentic-drift-detector\artifacts\Architecture_Overview.md"
pdf_path = r"c:\Users\acer\agentic-drift-detector\artifacts\Architecture_Overview.pdf"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

pdf = MarkdownPdf()
pdf.add_section(Section(text))
pdf.save(pdf_path)

print(f"PDF successfully generated at: {pdf_path}")
