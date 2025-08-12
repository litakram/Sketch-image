import streamlit as st

app_page=st.Page(page="views/main.py",
                 title="Generate",
                 icon="✨",
                 default=True)

documentation=st.Page(page="views/documentation.py",
                 title="Documentation",
                 icon=":material/description:")


pg=st.navigation(pages=[app_page,documentation,about_me])

pg.run()
