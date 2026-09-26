"""GICS hierarchy (2023 structure plus common legacy names): sub-industry -> industry -> sector.

Used to build the ``industry`` grouping between ``sector`` and ``subindustry``. Unknown sub-industries
fall back to being their own industry.
"""

from __future__ import annotations

INDUSTRIES: dict[str, list[str]] = {
    # Energy
    "Energy Equipment & Services": ["Oil & Gas Drilling", "Oil & Gas Equipment & Services"],
    "Oil, Gas & Consumable Fuels": ["Integrated Oil & Gas", "Oil & Gas Exploration & Production",
                                    "Oil & Gas Refining & Marketing", "Oil & Gas Storage & Transportation",
                                    "Coal & Consumable Fuels"],
    # Materials
    "Chemicals": ["Commodity Chemicals", "Diversified Chemicals", "Fertilizers & Agricultural Chemicals",
                  "Industrial Gases", "Specialty Chemicals"],
    "Construction Materials": ["Construction Materials"],
    "Containers & Packaging": ["Metal, Glass & Plastic Containers", "Metal & Glass Containers",
                               "Paper & Plastic Packaging Products & Materials", "Paper Packaging"],
    "Metals & Mining": ["Aluminum", "Diversified Metals & Mining", "Copper", "Gold", "Precious Metals & Minerals",
                        "Silver", "Steel"],
    "Paper & Forest Products": ["Forest Products", "Paper Products"],
    # Industrials
    "Aerospace & Defense": ["Aerospace & Defense"],
    "Building Products": ["Building Products"],
    "Construction & Engineering": ["Construction & Engineering"],
    "Electrical Equipment": ["Electrical Components & Equipment", "Heavy Electrical Equipment"],
    "Industrial Conglomerates": ["Industrial Conglomerates"],
    "Machinery": ["Construction Machinery & Heavy Transportation Equipment", "Construction Machinery & Heavy Trucks",
                  "Agricultural & Farm Machinery", "Industrial Machinery & Supplies & Components",
                  "Industrial Machinery"],
    "Trading Companies & Distributors": ["Trading Companies & Distributors"],
    "Commercial Services & Supplies": ["Commercial Printing", "Environmental & Facilities Services",
                                       "Office Services & Supplies", "Diversified Support Services",
                                       "Security & Alarm Services"],
    "Professional Services": ["Human Resource & Employment Services", "Research & Consulting Services",
                              "Data Processing & Outsourced Services"],
    "Air Freight & Logistics": ["Air Freight & Logistics"],
    "Passenger Airlines": ["Passenger Airlines", "Airlines"],
    "Marine Transportation": ["Marine Transportation", "Marine"],
    "Ground Transportation": ["Rail Transportation", "Railroads", "Cargo Ground Transportation", "Trucking",
                              "Passenger Ground Transportation"],
    "Transportation Infrastructure": ["Airport Services", "Highways & Railtracks", "Marine Ports & Services"],
    # Consumer Discretionary
    "Automobile Components": ["Automotive Parts & Equipment", "Tires & Rubber"],
    "Automobiles": ["Automobile Manufacturers", "Motorcycle Manufacturers"],
    "Household Durables": ["Consumer Electronics", "Home Furnishings", "Homebuilding", "Household Appliances",
                           "Housewares & Specialties"],
    "Leisure Products": ["Leisure Products"],
    "Textiles, Apparel & Luxury Goods": ["Apparel, Accessories & Luxury Goods", "Footwear", "Textiles"],
    "Hotels, Restaurants & Leisure": ["Casinos & Gaming", "Hotels, Resorts & Cruise Lines", "Leisure Facilities",
                                      "Restaurants"],
    "Diversified Consumer Services": ["Education Services", "Specialized Consumer Services"],
    "Distributors": ["Distributors"],
    "Broadline Retail": ["Broadline Retail", "Internet & Direct Marketing Retail", "General Merchandise Stores",
                         "Department Stores"],
    "Specialty Retail": ["Apparel Retail", "Computer & Electronics Retail", "Home Improvement Retail",
                         "Other Specialty Retail", "Specialty Stores", "Automotive Retail", "Homefurnishing Retail"],
    # Consumer Staples
    "Consumer Staples Distribution & Retail": ["Drug Retail", "Food Distributors", "Food Retail",
                                               "Consumer Staples Merchandise Retail", "Hypermarkets & Super Centers"],
    "Beverages": ["Brewers", "Distillers & Vintners", "Soft Drinks & Non-alcoholic Beverages", "Soft Drinks"],
    "Food Products": ["Agricultural Products & Services", "Agricultural Products", "Packaged Foods & Meats"],
    "Tobacco": ["Tobacco"],
    "Household Products": ["Household Products"],
    "Personal Care Products": ["Personal Care Products", "Personal Products"],
    # Health Care
    "Health Care Equipment & Supplies": ["Health Care Equipment", "Health Care Supplies"],
    "Health Care Providers & Services": ["Health Care Distributors", "Health Care Services", "Health Care Facilities",
                                         "Managed Health Care"],
    "Health Care Technology": ["Health Care Technology"],
    "Biotechnology": ["Biotechnology"],
    "Pharmaceuticals": ["Pharmaceuticals"],
    "Life Sciences Tools & Services": ["Life Sciences Tools & Services"],
    # Financials
    "Banks": ["Diversified Banks", "Regional Banks"],
    "Financial Services": ["Diversified Financial Services", "Other Diversified Financial Services",
                           "Multi-Sector Holdings", "Specialized Finance", "Commercial & Residential Mortgage Finance",
                           "Thrifts & Mortgage Finance", "Transaction & Payment Processing Services"],
    "Consumer Finance": ["Consumer Finance"],
    "Capital Markets": ["Asset Management & Custody Banks", "Investment Banking & Brokerage",
                        "Diversified Capital Markets", "Financial Exchanges & Data"],
    "Mortgage Real Estate Investment Trusts (REITs)": ["Mortgage REITs"],
    "Insurance": ["Insurance Brokers", "Life & Health Insurance", "Multi-line Insurance",
                  "Property & Casualty Insurance", "Reinsurance"],
    # Information Technology
    "IT Services": ["IT Consulting & Other Services", "Internet Services & Infrastructure"],
    "Software": ["Application Software", "Systems Software"],
    "Communications Equipment": ["Communications Equipment"],
    "Technology Hardware, Storage & Peripherals": ["Technology Hardware, Storage & Peripherals"],
    "Electronic Equipment, Instruments & Components": ["Electronic Equipment & Instruments", "Electronic Components",
                                                       "Electronic Manufacturing Services", "Technology Distributors"],
    "Semiconductors & Semiconductor Equipment": ["Semiconductor Materials & Equipment", "Semiconductor Equipment",
                                                 "Semiconductors"],
    # Communication Services
    "Diversified Telecommunication Services": ["Alternative Carriers", "Integrated Telecommunication Services"],
    "Wireless Telecommunication Services": ["Wireless Telecommunication Services"],
    "Media": ["Advertising", "Broadcasting", "Cable & Satellite", "Publishing"],
    "Entertainment": ["Movies & Entertainment", "Interactive Home Entertainment"],
    "Interactive Media & Services": ["Interactive Media & Services"],
    # Utilities
    "Electric Utilities": ["Electric Utilities"],
    "Gas Utilities": ["Gas Utilities"],
    "Multi-Utilities": ["Multi-Utilities"],
    "Water Utilities": ["Water Utilities"],
    "Independent Power and Renewable Electricity Producers": ["Independent Power Producers & Energy Traders",
                                                              "Renewable Electricity"],
    # Real Estate
    "Diversified REITs": ["Diversified REITs"],
    "Industrial REITs": ["Industrial REITs"],
    "Hotel & Resort REITs": ["Hotel & Resort REITs"],
    "Office REITs": ["Office REITs"],
    "Health Care REITs": ["Health Care REITs"],
    "Residential REITs": ["Multi-Family Residential REITs", "Single-Family Residential REITs", "Residential REITs"],
    "Retail REITs": ["Retail REITs"],
    "Specialized REITs": ["Other Specialized REITs", "Self-Storage REITs", "Telecom Tower REITs", "Timber REITs",
                          "Data Center REITs", "Specialized REITs"],
    "Real Estate Management & Development": ["Diversified Real Estate Activities", "Real Estate Operating Companies",
                                             "Real Estate Development", "Real Estate Services"],
}

SUB_TO_INDUSTRY: dict[str, str] = {s.lower(): ind for ind, subs in INDUSTRIES.items() for s in subs}


def industry_of(subindustry: str) -> str:
    return SUB_TO_INDUSTRY.get((subindustry or "").strip().lower(), subindustry or "Unknown")
