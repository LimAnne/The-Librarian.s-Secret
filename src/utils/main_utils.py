def process_llm_response(llm_response):
    sources_used = " \n".join(
        [
            source.metadata.get("source", "Unknown")
            .replace("\\", "/")
            .split("/")[-1][:-4]
            + " - page: "
            + str(source.metadata.get("page_label", source.metadata.get("page", "N/A")))
            for source in llm_response.get("source_documents", [])
        ]
    )

    ans = "\n\nSources: \n" + sources_used if sources_used else "\n\nNo sources found."
    return ans
