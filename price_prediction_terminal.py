"""
UK Supermarket Fiyat Tahmin Sistemi
Terminal Tabanlı Basit Arayüz
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Renk kodları (terminal için)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def load_data():
    """Model ve verileri yükle"""
    print_info("Model ve veriler yükleniyor...")
    
    try:
        model = joblib.load('models/linear_regression_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        df_cleaned = pd.read_csv('data/processed/cleaned_data.csv', parse_dates=['capture_date'])
        unique_products = pd.read_csv('data/processed/unique_products.csv')
        
        print_success("Model ve veriler başarıyla yüklendi!")
        return model, feature_names, df_cleaned, unique_products
    except Exception as e:
        print_error(f"Yükleme hatası: {str(e)}")
        return None, None, None, None

def get_user_choice(prompt, options, allow_search=False):
    """Kullanıcıdan seçim al"""
    print(f"\n{Colors.BOLD}{prompt}{Colors.ENDC}")
    
    if len(options) > 20 and allow_search:
        print_info(f"Toplam {len(options)} seçenek mevcut. Arama yapabilirsiniz.")
        search = input("🔍 Aramak için kelime girin (boş bırakın tüm listeyi görmek için): ").strip().lower()
        
        if search:
            filtered = [opt for opt in options if search in opt.lower()]
            if not filtered:
                print_warning("Arama sonucu bulunamadı. Tüm liste gösteriliyor.")
                filtered = options
            options = filtered
    
    # Sayfalama (20'şerli göster)
    page_size = 20
    page = 0
    
    while True:
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, len(options))
        
        print(f"\n{Colors.OKCYAN}[Sayfa {page + 1}/{(len(options) - 1) // page_size + 1}]{Colors.ENDC}")
        
        for i, option in enumerate(options[start_idx:end_idx], start=start_idx + 1):
            print(f"  {i}. {option}")
        
        if len(options) > page_size:
            print(f"\n  {Colors.WARNING}N: Sonraki sayfa | P: Önceki sayfa{Colors.ENDC}")
        
        choice = input(f"\n👉 Seçiminiz (1-{len(options)}): ").strip().upper()
        
        if choice == 'N' and end_idx < len(options):
            page += 1
            continue
        elif choice == 'P' and page > 0:
            page -= 1
            continue
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            else:
                print_error("Geçersiz seçim! Tekrar deneyin.")
        except ValueError:
            print_error("Lütfen bir sayı girin!")

def get_date_input():
    """Kullanıcıdan tarih al"""
    print(f"\n{Colors.BOLD}📅 Tahmin Tarihi Girin:{Colors.ENDC}")
    print_info("Format: GG/AA/YYYY (örn: 15/06/2024)")
    
    while True:
        date_str = input("👉 Tarih: ").strip()
        
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            return date_obj
        except ValueError:
            print_error("Geçersiz tarih formatı! GG/AA/YYYY formatında girin (örn: 15/06/2024)")

def prepare_features(selected_product, selected_supermarket, selected_category, 
                     prediction_date, df_cleaned, feature_names):
    """Tahmin için feature vektörü hazırla"""
    
    # Ürünün geçmiş verilerini bul
    product_data = df_cleaned[
        (df_cleaned['product_name'] == selected_product) & 
        (df_cleaned['supermarket_name'] == selected_supermarket) &
        (df_cleaned['category_name'] == selected_category)
    ]
    
    if len(product_data) == 0:
        return None, None
    
    latest_data = product_data.sort_values('capture_date').iloc[-1]
    
    # Tarih özellikleri
    month = prediction_date.month
    day = prediction_date.day
    day_of_week = prediction_date.weekday()
    week = prediction_date.isocalendar()[1]
    is_weekend = 1 if day_of_week >= 5 else 0
    is_month_start = 1 if day <= 7 else 0
    is_month_end = 1 if day >= 25 else 0
    
    # Sezon
    if month in [12, 1, 2]:
        season_encoded = 0
    elif month in [3, 4, 5]:
        season_encoded = 1
    elif month in [6, 7, 8]:
        season_encoded = 2
    else:
        season_encoded = 3
    
    # Supermarket one-hot encoding
    supermarket_features = {f'supermarket_{sm}': 0 for sm in ['ASDA', 'Aldi', 'Morrisons', 'Sains', 'Tesco']}
    if selected_supermarket == "Sainsbury's":
        supermarket_features['supermarket_Sains'] = 1
    else:
        supermarket_features[f'supermarket_{selected_supermarket}'] = 1
    
    # Category one-hot encoding
    category_features = {f'category_{cat}': 0 for cat in df_cleaned['category_name'].unique()}
    category_features[f'category_{selected_category}'] = 1
    
    # Diğer özellikler
    price_unit_gbp = latest_data['price_unit_gbp']
    
    unit_map = {'kg': 0, 'l': 1, 'unit': 2}
    unit_encoded = unit_map.get(latest_data['unit'], 2)
    
    price_cat_map = {'Ucuz': 2, 'Orta': 0, 'Pahalı': 1}
    price_category_encoded = price_cat_map.get(latest_data.get('price_category', 'Orta'), 0)
    
    is_own_brand = latest_data.get('is_own_brand', 0)
    
    price_to_unit_ratio = product_data['price_gbp'].mean() / (product_data['price_unit_gbp'].mean() + 0.001)
    price_vs_category_avg = 0
    price_vs_supermarket_avg = 0
    
    is_premium_category = 1 if selected_category in ['health_products', 'baby_products', 'home'] else 0
    is_discount_supermarket = 1 if selected_supermarket in ['Aldi', 'ASDA'] else 0
    premium_category_x_premium_supermarket = is_premium_category * (1 - is_discount_supermarket)
    
    # Feature dict
    feature_dict = {
        'price_unit_gbp': price_unit_gbp,
        **supermarket_features,
        **category_features,
        'unit_encoded': unit_encoded,
        'price_category_encoded': price_category_encoded,
        'is_own_brand': is_own_brand,
        'month': month,
        'day': day,
        'day_of_week': day_of_week,
        'week': week,
        'is_weekend': is_weekend,
        'price_to_unit_ratio': price_to_unit_ratio,
        'price_vs_category_avg': price_vs_category_avg,
        'price_vs_supermarket_avg': price_vs_supermarket_avg,
        'is_month_start': is_month_start,
        'is_month_end': is_month_end,
        'season_encoded': season_encoded,
        'is_premium_category': is_premium_category,
        'is_discount_supermarket': is_discount_supermarket,
        'premium_category_x_premium_supermarket': premium_category_x_premium_supermarket
    }
    
    X_pred = pd.DataFrame([feature_dict])
    X_pred = X_pred[feature_names]
    
    return X_pred, product_data

def main():
    """Ana program"""
    print_header("🛒 UK SUPERMARKET FİYAT TAHMİN SİSTEMİ")
    print_info("Linear Regression Model (R²=99.86%)")
    
    # Verileri yükle
    model, feature_names, df_cleaned, unique_products = load_data()
    
    if model is None:
        return
    
    print_success(f"Toplam {len(unique_products):,} benzersiz ürün yüklendi")
    print_success(f"Tarih aralığı: {df_cleaned['capture_date'].min().strftime('%d/%m/%Y')} - {df_cleaned['capture_date'].max().strftime('%d/%m/%Y')}")
    
    while True:
        # 1. Supermarket seçimi
        supermarkets = sorted(df_cleaned['supermarket_name'].unique().tolist())
        selected_supermarket = get_user_choice("🏪 SÜPERMARKET SEÇİMİ", supermarkets)
        print_success(f"Seçilen: {selected_supermarket}")
        
        # 2. Kategori seçimi
        categories = sorted(df_cleaned[df_cleaned['supermarket_name'] == selected_supermarket]['category_name'].unique().tolist())
        selected_category = get_user_choice("📦 KATEGORİ SEÇİMİ", categories)
        print_success(f"Seçilen: {selected_category}")
        
        # 3. Ürün seçimi
        filtered_products = unique_products[
            (unique_products['supermarket_name'] == selected_supermarket) & 
            (unique_products['category_name'] == selected_category)
        ]['product_name'].sort_values().unique().tolist()
        
        if not filtered_products:
            print_warning("Bu kombinasyon için ürün bulunamadı!")
            continue
        
        selected_product = get_user_choice("🛍️  ÜRÜN SEÇİMİ", filtered_products, allow_search=True)
        print_success(f"Seçilen: {selected_product}")
        
        # 4. Tarih seçimi
        prediction_date = get_date_input()
        print_success(f"Seçilen: {prediction_date.strftime('%d/%m/%Y')}")
        
        # 5. Tahmin yap
        print_info("\n🎯 Tahmin yapılıyor...")
        
        X_pred, product_data = prepare_features(
            selected_product, selected_supermarket, selected_category,
            prediction_date, df_cleaned, feature_names
        )
        
        if X_pred is None:
            print_error("Bu ürün için geçmiş veri bulunamadı!")
            continue
        
        # Tahmin
        predicted_price = model.predict(X_pred)[0]
        
        # Ölçeklendirme düzeltmesi
        actual_avg_price = product_data['price_gbp'].mean()
        actual_std_price = product_data['price_gbp'].std()
        final_predicted_price = predicted_price * actual_std_price + actual_avg_price
        final_predicted_price = max(0.01, final_predicted_price)
        
        # Sonuçları göster
        print_header("💰 TAHMİN SONUCU")
        
        print(f"{Colors.BOLD}{Colors.OKGREEN}Tahmin Edilen Fiyat: £{final_predicted_price:.2f}{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}📊 İSTATİSTİKLER:{Colors.ENDC}")
        print(f"  • Ortalama Fiyat: £{actual_avg_price:.2f}")
        print(f"  • En Düşük Fiyat: £{product_data['price_gbp'].min():.2f}")
        print(f"  • En Yüksek Fiyat: £{product_data['price_gbp'].max():.2f}")
        print(f"  • Standart Sapma: £{actual_std_price:.2f}")
        print(f"  • Veri Sayısı: {len(product_data)} kayıt")
        
        print(f"\n{Colors.BOLD}📝 ÜRÜN BİLGİLERİ:{Colors.ENDC}")
        print(f"  • Ürün: {selected_product}")
        print(f"  • Market: {selected_supermarket}")
        print(f"  • Kategori: {selected_category}")
        print(f"  • Tahmin Tarihi: {prediction_date.strftime('%d %B %Y')}")
        
        # Tekrar tahmin yap?
        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
        again = input("\n🔄 Başka bir tahmin yapmak ister misiniz? (E/H): ").strip().upper()
        
        if again != 'E':
            print_header("👋 GÜLE GÜLE!")
            print_success("Program sonlandırıldı.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_header("👋 PROGRAM SONLANDIRILDI")
        print_info("Kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print_error(f"Beklenmeyen hata: {str(e)}")
