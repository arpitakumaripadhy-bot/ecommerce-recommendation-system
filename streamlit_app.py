import streamlit as st

from src.recommender import (
    load_recommendation_models,
    recommend_products
)


st.set_page_config(
    page_title="E-commerce Recommendation System",
    page_icon="🛍️",
    layout="wide"
)


@st.cache_resource
def load_models():
    return load_recommendation_models()


def main():
    st.title("🛍️ E-commerce Recommendation System")

    st.write(
        "Discover personalized product recommendations "
        "using item-based collaborative filtering."
    )

    st.divider()

    try:
        popularity_model, collaborative_model = load_models()

    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Open your project terminal and run "
            "`python -u train_pipeline.py` first."
        )
        st.stop()

    customer_ids = collaborative_model["customer_ids"]

    if not customer_ids:
        st.warning("No customers are available.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.selectbox(
            "Select Customer ID",
            options=customer_ids
        )

    with col2:
        top_n = st.number_input(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )

    if st.button("Get Recommendations", type="primary"):
        recommendations = recommend_products(
            customer_id=customer_id,
            top_n=int(top_n),
            popularity_model=popularity_model,
            collaborative_model=collaborative_model
        )

        st.subheader("Recommended Products")

        if recommendations.empty:
            st.warning(
                "No unseen products are available to recommend "
                "for this customer."
            )
        else:
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Recommendation scores estimate relevance from "
                "past ratings and item similarities. They are "
                "not guaranteed future ratings."
            )

            csv_data = recommendations.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Recommendations as CSV",
                data=csv_data,
                file_name=f"recommendations_{customer_id}.csv",
                mime="text/csv"
            )

    st.divider()

    st.subheader("About This Project")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customers",
            len(collaborative_model["customer_ids"])
        )

    with col2:
        st.metric(
            "Products",
            len(collaborative_model["product_ids"])
        )

    with col3:
        st.metric(
            "Recommendation Method",
            "Item-based CF"
        )

    st.write(
        "The system recommends products based on similarities "
        "between items and a customer's historical ratings. "
        "Popularity-based recommendations are used as a fallback."
    )


if __name__ == "__main__":
    main()