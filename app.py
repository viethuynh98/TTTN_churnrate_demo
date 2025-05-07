import streamlit as st

# Initialize connection.
conn = st.connection('mysql', type='sql')

# Perform query.
df = conn.query('SELECT * from city;', ttl=600)

# Print results.
for row in df.itertuples():
    st.write(f"{row.Name} has a :{row.Population}:")