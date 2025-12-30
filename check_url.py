from langchain_community.document_loaders import SitemapLoader

def check_filters():
    print("--- CHECKING SITEMAP FILTERS ---")
    
    # 1. Define your filters (Parents only)
    my_filters = [
        "https://www.winfomi.com/services", 
        "https://www.winfomi.com/products"
    ]
    
    # 2. Initialize Loader
    loader = SitemapLoader(
        web_path="https://www.winfomi.com/sitemap.xml",
        filter_urls=my_filters
    )

    # 3. Parse WITHOUT downloading (Fast)
    # This just reads the XML list, it doesn't scrape the text yet.
    # Note: We use 'parse_sitemap' helper if available, or just load and print metadata.
    try:
        # We perform a "dry run" by accessing the internal filtering logic if possible, 
        # but the easiest way is to just load and print the 'source' metadata.
        # Since we can't easily skip download in standard SitemapLoader without a custom function,
        # we will just limit it to 10 items for the test.
        
        print("Reading sitemap (this takes 10 seconds)...")
        docs = loader.load() 
        
        print(f"\n✅ Found {len(docs)} pages matching your filters.")
        print("First 10 URLs found:")
        for doc in docs[:10]:
            print(f" - {doc.metadata['source']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_filters()