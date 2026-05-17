#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::filesystem;

// --- 1. TEMEL VERİ YAPILARI ---

struct MyImage
{
    int width;
    int height;
    vector<vector<double>> data;

    MyImage(int w, int h) : width(w), height(h)
    {
        data.resize(h, vector<double>(w, 0.0));
    }
};

struct RGBPixel
{
    uchar b, g, r;
};

struct MyColorImage
{
    int width;
    int height;
    vector<vector<RGBPixel>> data;

    MyColorImage(int w, int h) : width(w), height(h)
    {
        data.resize(h, vector<RGBPixel>(w, {0, 0, 0}));
    }
};

struct Point2D
{
    int x, y;
    double score;
};

struct Match
{
    Point2D pt1;
    Point2D pt2;
    double distance;
};

struct RansacResult
{
    vector<vector<double>> H;
    vector<Match> inliers;
};

// --- KÖPRÜ VE GÖRSELLEŞTİRME FONKSİYONLARI ---
MyImage convertCvMatToMyImage(const cv::Mat &cv_img)
{
    cv::Mat gray_img;
    if (cv_img.channels() == 3)
        cv::cvtColor(cv_img, gray_img, cv::COLOR_BGR2GRAY);
    else
        gray_img = cv_img;

    MyImage my_img(gray_img.cols, gray_img.rows);
    for (int y = 0; y < gray_img.rows; y++)
    {
        for (int x = 0; x < gray_img.cols; x++)
        {
            my_img.data[y][x] = (double)gray_img.at<uchar>(y, x);
        }
    }
    return my_img;
}

MyColorImage convertCvMatToMyColorImage(const cv::Mat &cv_img)
{
    MyColorImage my_img(cv_img.cols, cv_img.rows);
    for (int y = 0; y < cv_img.rows; y++)
    {
        for (int x = 0; x < cv_img.cols; x++)
        {
            cv::Vec3b p = cv_img.at<cv::Vec3b>(y, x);
            my_img.data[y][x] = {p[0], p[1], p[2]};
        }
    }
    return my_img;
}

cv::Mat convertMyColorImageToCvMat(const MyColorImage &my_img)
{
    cv::Mat cv_img(my_img.height, my_img.width, CV_8UC3);
    for (int y = 0; y < my_img.height; y++)
    {
        for (int x = 0; x < my_img.width; x++)
        {
            cv_img.at<cv::Vec3b>(y, x) = cv::Vec3b(my_img.data[y][x].b, my_img.data[y][x].g, my_img.data[y][x].r);
        }
    }
    return cv_img;
}

// Raporda kullanmak üzere köşe noktalarını resme çizip kaydeder
void saveHarrisCornersImage(const cv::Mat &img_cv, const vector<Point2D> &corners, const string &filename)
{
    cv::Mat outImg = img_cv.clone();
    for (const auto &pt : corners)
    {
        cv::circle(outImg, cv::Point(pt.x, pt.y), 2, cv::Scalar(0, 0, 255), -1); // Kırmızı Noktalar
    }
    cv::imwrite(filename, outImg);
}

void saveMatchesImage(const cv::Mat &img1_cv, const cv::Mat &img2_cv, const vector<Match> &matches, const string &filename)
{
    cv::Mat c1, c2;
    if (img1_cv.channels() == 1)
        cv::cvtColor(img1_cv, c1, cv::COLOR_GRAY2BGR);
    else
        c1 = img1_cv.clone();
    if (img2_cv.channels() == 1)
        cv::cvtColor(img2_cv, c2, cv::COLOR_GRAY2BGR);
    else
        c2 = img2_cv.clone();

    int max_rows = max(c1.rows, c2.rows);
    int total_cols = c1.cols + c2.cols;
    cv::Mat outImg = cv::Mat::zeros(max_rows, total_cols, CV_8UC3);

    c1.copyTo(outImg(cv::Rect(0, 0, c1.cols, c1.rows)));
    c2.copyTo(outImg(cv::Rect(c1.cols, 0, c2.cols, c2.rows)));

    for (auto &m : matches)
    {
        cv::Point p1(m.pt1.x, m.pt1.y);
        cv::Point p2(m.pt2.x + c1.cols, m.pt2.y);
        cv::line(outImg, p1, p2, cv::Scalar(0, 255, 0), 1);
        cv::circle(outImg, p1, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(outImg, p2, 3, cv::Scalar(0, 0, 255), -1);
    }
    cv::imwrite(filename, outImg);
}

// --- 2. HARRIS KÖŞE TESPİTİ VE ÖLÇEK (SCALE) BAĞIMSIZLIĞI ---
vector<Point2D> detectHarrisCorners(const MyImage &img, int window_size = 5)
{
    int w = img.width, h = img.height;
    MyImage Ix2(w, h), Iy2(w, h), Ixy(w, h);
    for (int y = 1; y < h - 1; y++)
    {
        for (int x = 1; x < w - 1; x++)
        {
            double Ix = (img.data[y - 1][x + 1] + 2 * img.data[y][x + 1] + img.data[y + 1][x + 1]) -
                        (img.data[y - 1][x - 1] + 2 * img.data[y][x - 1] + img.data[y + 1][x - 1]);
            double Iy = (img.data[y + 1][x - 1] + 2 * img.data[y + 1][x] + img.data[y + 1][x + 1]) -
                        (img.data[y - 1][x - 1] + 2 * img.data[y - 1][x] + img.data[y - 1][x + 1]);
            Ix2.data[y][x] = Ix * Ix;
            Iy2.data[y][x] = Iy * Iy;
            Ixy.data[y][x] = Ix * Iy;
        }
    }

    MyImage R_scores(w, h);
    double R_max = 0.0, k = 0.04;
    int offset = window_size / 2;
    for (int y = offset + 1; y < h - offset - 1; y++)
    {
        for (int x = offset + 1; x < w - offset - 1; x++)
        {
            double Sxx = 0, Syy = 0, Sxy = 0;
            for (int wy = -offset; wy <= offset; wy++)
            {
                for (int wx = -offset; wx <= offset; wx++)
                {
                    Sxx += Ix2.data[y + wy][x + wx];
                    Syy += Iy2.data[y + wy][x + wx];
                    Sxy += Ixy.data[y + wy][x + wx];
                }
            }
            double R = (Sxx * Syy) - (Sxy * Sxy) - k * ((Sxx + Syy) * (Sxx + Syy));
            R_scores.data[y][x] = R;
            if (R > R_max)
                R_max = R;
        }
    }

    double dynamic_threshold = 0.01 * R_max;
    vector<Point2D> corners;

    for (int y = offset + 1; y < h - offset - 1; y++)
    {
        for (int x = offset + 1; x < w - offset - 1; x++)
        {
            double current_score = R_scores.data[y][x];
            if (current_score > dynamic_threshold)
            {
                bool is_max = true;
                for (int wy = -offset; wy <= offset; wy++)
                {
                    for (int wx = -offset; wx <= offset; wx++)
                    {
                        if (wy == 0 && wx == 0)
                            continue;
                        if (R_scores.data[y + wy][x + wx] > current_score)
                        {
                            is_max = false;
                            break;
                        }
                    }
                    if (!is_max)
                        break;
                }
                if (is_max)
                    corners.push_back({x, y, current_score});
            }
        }
    }
    return corners;
}

MyImage downsampleImage(const MyImage &src)
{
    int new_w = src.width / 2, new_h = src.height / 2;
    MyImage dst(new_w, new_h);
    for (int y = 0; y < new_h; y++)
    {
        for (int x = 0; x < new_w; x++)
        {
            dst.data[y][x] = (src.data[y * 2][x * 2] + src.data[y * 2][x * 2 + 1] +
                              src.data[y * 2 + 1][x * 2] + src.data[y * 2 + 1][x * 2 + 1]) /
                             4.0;
        }
    }
    return dst;
}

vector<Point2D> detectMultiScaleHarris(const MyImage &img)
{
    vector<Point2D> all_corners;
    vector<Point2D> corners_lvl1 = detectHarrisCorners(img);
    all_corners.insert(all_corners.end(), corners_lvl1.begin(), corners_lvl1.end());

    if (img.width > 40 && img.height > 40)
    {
        MyImage img_half = downsampleImage(img);
        vector<Point2D> corners_lvl2 = detectHarrisCorners(img_half);
        for (auto &pt : corners_lvl2)
        {
            pt.x *= 2;
            pt.y *= 2;
            all_corners.push_back(pt);
        }

        MyImage img_quarter = downsampleImage(img_half);
        vector<Point2D> corners_lvl3 = detectHarrisCorners(img_quarter);
        for (auto &pt : corners_lvl3)
        {
            pt.x *= 4;
            pt.y *= 4;
            all_corners.push_back(pt);
        }
    }
    return all_corners;
}

// --- 3. ÖZELLİK EŞLEŞTİRME ---
vector<Match> matchFeatures(const MyImage &img1, const vector<Point2D> &corners1,
                            const MyImage &img2, const vector<Point2D> &corners2)
{
    vector<Match> matches;
    int offset = 4;

    for (size_t i = 0; i < corners1.size(); i++)
    {
        int x1 = corners1[i].x, y1 = corners1[i].y;
        if (x1 < offset || x1 >= img1.width - offset || y1 < offset || y1 >= img1.height - offset)
            continue;

        double best_dist = 1e9, second_best_dist = 1e9;
        int best_match_idx = -1;

        for (size_t j = 0; j < corners2.size(); j++)
        {
            int x2 = corners2[j].x, y2 = corners2[j].y;
            if (x2 < offset || x2 >= img2.width - offset || y2 < offset || y2 >= img2.height - offset)
                continue;

            double sad_dist = 0.0;
            for (int wy = -offset; wy <= offset; wy++)
            {
                for (int wx = -offset; wx <= offset; wx++)
                {
                    sad_dist += std::abs(img1.data[y1 + wy][x1 + wx] - img2.data[y2 + wy][x2 + wx]);
                }
            }

            if (sad_dist < best_dist)
            {
                second_best_dist = best_dist;
                best_dist = sad_dist;
                best_match_idx = j;
            }
            else if (sad_dist < second_best_dist)
            {
                second_best_dist = sad_dist;
            }
        }

        if (best_match_idx != -1 && best_dist < 0.8 * second_best_dist)
        {
            matches.push_back({corners1[i], corners2[best_match_idx], best_dist});
        }
    }
    return matches;
}

// --- 4. RANSAC VE HOMOGRAFİ (ESNEMEYİ ÖNLEYEN PURE TRANSLATION MODELİ) ---
RansacResult computeHomographyRANSAC(const vector<Match> &matches)
{
    RansacResult result;
    result.H = vector<vector<double>>(3, vector<double>(3, 0.0));
    int num_iterations = 5000; // Sadece kaydırma aradığımız için 5k iterasyon fazlasıyla yeterli
    double delta_threshold = 5.0;
    int best_inlier_count = 0;

    if (matches.empty())
        return result;

    for (int iter = 0; iter < num_iterations; iter++)
    {
        // Perspektif bükme için 4 nokta lazımdı, sadece kaydırma (translation) için 1 nokta yeterli!
        Match sample = matches[rand() % matches.size()];

        // İki resim arasındaki X ve Y eksenindeki ham kaydırma miktarını buluyoruz
        double tx = sample.pt2.x - sample.pt1.x;
        double ty = sample.pt2.y - sample.pt1.y;

        // Saf kaydırma matrisini inşa et (Dönme yok, esneme yok, sadece öteleme)
        vector<vector<double>> H_iter(3, vector<double>(3, 0.0));
        H_iter[0][0] = 1.0;
        H_iter[0][1] = 0.0;
        H_iter[0][2] = tx;
        H_iter[1][0] = 0.0;
        H_iter[1][1] = 1.0;
        H_iter[1][2] = ty;
        H_iter[2][0] = 0.0;
        H_iter[2][1] = 0.0;
        H_iter[2][2] = 1.0;

        vector<Match> current_inliers;
        for (size_t i = 0; i < matches.size(); i++)
        {
            // Bu kaydırma miktarına göre 1. resimdeki noktanın gitmesi gereken yer:
            double xp_est = matches[i].pt1.x + tx;
            double yp_est = matches[i].pt1.y + ty;

            // Gerçek konum ile tahmin edilen konum arasındaki mesafeyi ölç
            double dist = sqrt(pow(xp_est - matches[i].pt2.x, 2) + pow(yp_est - matches[i].pt2.y, 2));
            if (dist < delta_threshold)
                current_inliers.push_back(matches[i]);
        }

        if (current_inliers.size() > best_inlier_count)
        {
            best_inlier_count = current_inliers.size();
            result.H = H_iter;
            result.inliers = current_inliers;
        }
    }
    // --- YENİ DÜZELTME: EN İYİ INLIER'LARIN ORTALAMASINI ALARAK GHOSTING'İ SIFIRLA ---
    if (best_inlier_count > 0)
    {
        double total_tx = 0.0;
        double total_ty = 0.0;
        for (const auto &inlier : result.inliers)
        {
            total_tx += (inlier.pt2.x - inlier.pt1.x);
            total_ty += (inlier.pt2.y - inlier.pt1.y);
        }
        // Tam sayı yerine double hassasiyetinde alt-piksel kayması hesaplanıyor
        double final_tx = total_tx / best_inlier_count;
        double final_ty = total_ty / best_inlier_count;

        result.H[0][0] = 1.0;
        result.H[0][1] = 0.0;
        result.H[0][2] = final_tx;
        result.H[1][0] = 0.0;
        result.H[1][1] = 1.0;
        result.H[1][2] = final_ty;
        result.H[2][0] = 0.0;
        result.H[2][1] = 0.0;
        result.H[2][2] = 1.0;
    }
    return result;
}

// --- 5. MANUEL WARPING VE BLENDING ---
vector<vector<double>> invertMatrix3x3(const vector<vector<double>> &M)
{
    double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

    vector<vector<double>> inv(3, vector<double>(3, 0.0));
    if (abs(det) < 1e-9)
        return inv;

    double invdet = 1.0 / det;
    inv[0][0] = (M[1][1] * M[2][2] - M[2][1] * M[1][2]) * invdet;
    inv[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invdet;
    inv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invdet;
    inv[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invdet;
    inv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invdet;
    inv[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * invdet;
    inv[2][0] = (M[1][0] * M[2][1] - M[2][0] * M[1][1]) * invdet;
    inv[2][1] = (M[2][0] * M[0][1] - M[0][0] * M[2][1]) * invdet;
    inv[2][2] = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * invdet;

    return inv;
}

// --- ANA FONKSİYON ---
int main()
{
    srand(time(0));
    string base_input_dir = "input";
    if (!fs::exists(base_input_dir) || !fs::is_directory(base_input_dir))
    {
        cout << "Hata: Lutfen 'input' adinda bir klasor acip icine veri setlerini ekleyin." << endl;
        return -1;
    }

    // Input klasöründeki alt klasörleri bul
    vector<string> available_folders;
    for (const auto &entry : fs::directory_iterator(base_input_dir))
    {
        if (entry.is_directory())
        {
            available_folders.push_back(entry.path().filename().string());
        }
    }

    if (available_folders.empty())
    {
        cout << "Hata: 'input' klasorunun icinde hic alt klasor (veri seti) bulunamadi!" << endl;
        return -1;
    }

    sort(available_folders.begin(), available_folders.end());

    // KULLANICIYA INTERAKTIF MENU SUNALIM
    cout << "\n=======================================================" << endl;
    cout << "     PANORAMA OLUSTURUCU MENU (RAPOR CIKTILI MOD)" << endl;
    cout << "=======================================================" << endl;
    cout << "  0 : Hepsini Sirayla Islet" << endl;
    for (size_t i = 0; i < available_folders.size(); i++)
    {
        cout << "  " << i + 1 << " : Sadece '" << available_folders[i] << "' klasorunu islet" << endl;
    }
    cout << "=======================================================" << endl;

    int choice;
    cout << "Lutfen bir islem numarasi secin: ";
    if (!(cin >> choice))
    {
        cout << "Hatali giris yaptiniz! Program sonlandiriliyor." << endl;
        return -1;
    }

    vector<string> folders_to_process;
    if (choice == 0)
    {
        folders_to_process = available_folders;
        cout << "Bilgi: Tum klasorler isleme alindi!" << endl;
    }
    else if (choice > 0 && choice <= (int)available_folders.size())
    {
        folders_to_process.push_back(available_folders[choice - 1]);
        cout << "Bilgi: Sadece '" << available_folders[choice - 1] << "' isleme alindi!" << endl;
    }
    else
    {
        cout << "Gecersiz bir secim yaptiniz! Program sonlandiriliyor." << endl;
        return -1;
    }

    // Ana Sonuç Klasörünü Oluştur
    string base_output_dir = "result";
    if (!fs::exists(base_output_dir))
        fs::create_directory(base_output_dir);

    // SADECE SEÇİLEN KLASÖRLERİ İŞLE
    for (const string &folder_name : folders_to_process)
    {
        string folder_path = base_input_dir + "/" + folder_name;

        // Alt Sonuç Klasörünü Oluştur (Örn: result/hill)
        string out_dir = base_output_dir + "/" + folder_name;
        if (!fs::exists(out_dir))
            fs::create_directories(out_dir);

        cout << "\n=======================================================" << endl;
        cout << ">>> BASLIYOR: '" << folder_name << "' klasoru (RAW / HAM SONUC ODAKLI MOD) " << endl;
        cout << "=======================================================" << endl;

        vector<string> image_files;
        for (const auto &img_entry : fs::directory_iterator(folder_path))
        {
            string ext = img_entry.path().extension().string();
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
                image_files.push_back(img_entry.path().string());
        }
        sort(image_files.begin(), image_files.end());

        if (image_files.size() < 2)
        {
            cout << "Uyari: '" << folder_name << "' icinde yeterli fotograf yok, atlaniyor." << endl;
            continue;
        }

        vector<cv::Mat> images_color_cv;
        vector<MyImage> images_gray_my;
        vector<vector<Point2D>> all_corners;

        // ADIM 1: Resimleri Oku, Köşeleri Bul ve Kaydet
        for (size_t i = 0; i < image_files.size(); i++)
        {
            cv::Mat img = cv::imread(image_files[i], cv::IMREAD_COLOR);
            images_color_cv.push_back(img);

            MyImage gray_img = convertCvMatToMyImage(img);
            images_gray_my.push_back(gray_img);

            // Harris köşelerini tespit et ve klasöre kaydet
            vector<Point2D> corners = detectMultiScaleHarris(gray_img);
            all_corners.push_back(corners);
            saveHarrisCornersImage(img, corners, out_dir + "/Adim1_Harris_Koseler_" + to_string(i + 1) + ".jpg");
        }

        vector<cv::Mat> H_pairwise;
        for (size_t i = 1; i < images_gray_my.size(); i++)
        {
            cout << "--- " << i << ".jpg ile " << i + 1 << ".jpg eslestiriliyor (Ratio Test & RANSAC 10k iter) ---" << endl;

            // Eşleşmeleri bul ve RANSAC ile homografiyi hesapla
            vector<Match> matches = matchFeatures(images_gray_my[i - 1], all_corners[i - 1], images_gray_my[i], all_corners[i]);
            RansacResult ransac_out = computeHomographyRANSAC(matches);

            // ADIM 2: Eşleşmeleri klasöre kaydet
            saveMatchesImage(images_color_cv[i - 1], images_color_cv[i], ransac_out.inliers, out_dir + "/Adim2_Eslesmeler_" + to_string(i) + "-" + to_string(i + 1) + ".jpg");

            cv::Mat H(3, 3, CV_64F);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    H.at<double>(r, c) = ransac_out.H[r][c];
            H_pairwise.push_back(H);
        }

        // Adım 3: Global Homografileri Hesapla
        vector<cv::Mat> H_global;
        H_global.push_back(cv::Mat::eye(3, 3, CV_64F)); // H_0 = I
        for (size_t i = 0; i < H_pairwise.size(); i++)
        {
            cv::Mat H_inv_step = H_pairwise[i].inv();
            H_global.push_back(H_global[i] * H_inv_step);
        }

        // REFERANSI ORTAYA KAYDIRMA (Sünmeyi azaltmak için)
        int middle_idx = images_gray_my.size() / 2;
        cv::Mat H_shift = H_global[middle_idx].inv();

        for (size_t i = 0; i < H_global.size(); i++)
        {
            H_global[i] = H_shift * H_global[i];
        }

        // Adım 4: Tuval (Canvas) Boyutunu ve Sınırlarını Bul
        double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
        for (size_t i = 0; i < images_gray_my.size(); i++)
        {
            vector<cv::Point2f> corners = {
                cv::Point2f(0, 0),
                cv::Point2f(images_gray_my[i].width, 0),
                cv::Point2f(0, images_gray_my[i].height),
                cv::Point2f(images_gray_my[i].width, images_gray_my[i].height)};
            vector<cv::Point2f> corners_warped;
            cv::perspectiveTransform(corners, corners_warped, H_global[i]);
            for (auto &pt : corners_warped)
            {
                min_x = min(min_x, (double)pt.x);
                max_x = max(max_x, (double)pt.x);
                min_y = min(min_y, (double)pt.y);
                max_y = max(max_y, (double)pt.y);
            }
        }

        // RAM GÜVENLİK KONTROLÜ
        int canvas_w = round(max_x - min_x);
        int canvas_h = round(max_y - min_y);
        cout << ">>> Hesaplan Tuval Boyutu: " << canvas_w << " x " << canvas_h << " piksel." << endl;

        if (canvas_w > 15000 || canvas_h > 15000 || canvas_w <= 0 || canvas_h <= 0)
        {
            cout << "HATA: Uzama cok fazla! RAM patlamamasi icin bu islem atliyor." << endl;
            continue;
        }

        int shift_x = -round(min_x), shift_y = -round(min_y);
        cv::Size canvas_size(canvas_w, canvas_h);
        cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, shift_x, 0, 1, shift_y, 0, 0, 1);

        // Adım 5 ve 6: Tuvali Oluştur, Manuel Eğ ve Tüy Gibi Birleştir (Feathering Blending)
        cout << ">>> Pikseller agirlikli ortalama (Feathering) ile puruzsuz birlestiriliyor..." << endl;

        vector<vector<double>> sum_r(canvas_size.height, vector<double>(canvas_size.width, 0.0));
        vector<vector<double>> sum_g(canvas_size.height, vector<double>(canvas_size.width, 0.0));
        vector<vector<double>> sum_b(canvas_size.height, vector<double>(canvas_size.width, 0.0));
        vector<vector<double>> sum_w(canvas_size.height, vector<double>(canvas_size.width, 0.0));

        for (size_t i = 0; i < images_color_cv.size(); i++)
        {
            cv::Mat H_final_cv = T * H_global[i];
            vector<vector<double>> H_final(3, vector<double>(3));
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    H_final[r][c] = H_final_cv.at<double>(r, c);

            vector<vector<double>> H_inv = invertMatrix3x3(H_final);
            MyColorImage src_my = convertCvMatToMyColorImage(images_color_cv[i]);

            for (int y = 0; y < canvas_size.height; y++)
            {
                for (int x = 0; x < canvas_size.width; x++)
                {
                    double src_x_w = H_inv[0][0] * x + H_inv[0][1] * y + H_inv[0][2];
                    double src_y_w = H_inv[1][0] * x + H_inv[1][1] * y + H_inv[1][2];
                    double w = H_inv[2][0] * x + H_inv[2][1] * y + H_inv[2][2];

                    if (abs(w) > 1e-6)
                    {

                        int src_x = round(src_x_w / w);
                        int src_y = round(src_y_w / w);

                        if (src_x >= 0 && src_x < src_my.width && src_y >= 0 && src_y < src_my.height)
                        {
                            double dist_x = min(src_x, src_my.width - 1 - src_x);
                            double dist_y = min(src_y, src_my.height - 1 - src_y);
                            double weight = min(dist_x, dist_y) + 1.0;

                            RGBPixel p = src_my.data[src_y][src_x];

                            sum_r[y][x] += p.r * weight;
                            sum_g[y][x] += p.g * weight;
                            sum_b[y][x] += p.b * weight;
                            sum_w[y][x] += weight;
                        }
                    }
                }
            }
        }

        // Doğrudan Sonuç Tuvalini Çıktı Olarak Al (Kırpma İşlemi Yok)
        MyColorImage result_pano_my(canvas_size.width, canvas_size.height);
        for (int y = 0; y < canvas_size.height; y++)
        {
            for (int x = 0; x < canvas_size.width; x++)
            {
                if (sum_w[y][x] > 0)
                {
                    result_pano_my.data[y][x].r = static_cast<uchar>(sum_r[y][x] / sum_w[y][x]);
                    result_pano_my.data[y][x].g = static_cast<uchar>(sum_g[y][x] / sum_w[y][x]);
                    result_pano_my.data[y][x].b = static_cast<uchar>(sum_b[y][x] / sum_w[y][x]);
                }
            }
        }

        cout << ">>> Ham sonuc disari aktariliyor (KIRPMA IPTAL)..." << endl;

        // ADIM 3: Final Panorama'yı result/klasor_adi içine kaydet
        string output_filename = out_dir + "/Adim3_Final_Panorama.jpg";
        cv::Mat final_panorama_cv = convertMyColorImageToCvMat(result_pano_my);
        cv::imwrite(output_filename, final_panorama_cv);

        cout << ">>> TAMAMLANDI! Sonuclar '" << out_dir << "' icine kaydedildi." << endl;
    }

    return 0;
}