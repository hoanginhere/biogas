from django.urls import path
from .views import *

urlpatterns = [
    path('login/',login_current_user),
    path('logout/',logout_current_user),
    path('register/',register_new_user),
    path('edit_profile/',edit_user_profile),
    path('reg_code/',generate_registration_code),
    path('verify/',verify_registration),
    path('profile_view/',ProfileListView)
]