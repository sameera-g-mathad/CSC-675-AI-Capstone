import { KeyboardAvoidingView, Platform, SafeAreaView, StyleSheet, Text, View } from "react-native";
import React, { useEffect } from "react";
import { useNavigation, useLocalSearchParams, useRouter } from "expo-router";
import Chatbox from "@/app/_components/Chatbox";
import Content from "@/app/_components/Content";
import { HeaderBackButton } from '@react-navigation/elements';

export default function Chat() {
    const navigation = useNavigation();
    const router = useRouter()
    const { title } = useLocalSearchParams();
    useEffect(() => {
        navigation.setOptions({
            header: () =>
            (
                <View style={{
                    height: 70,
                    padding: 10,
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    backgroundColor: '#6ee7b7'
                }}>
                    <HeaderBackButton onPress={() => router.replace('/')} />
                    < Text >{title}</ Text>
                </View>
            )
        })
    }, [])
    return (
        <SafeAreaView style={styles.viewContainer}>
            <KeyboardAvoidingView
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                keyboardVerticalOffset={130}
                style={{ flex: 1 }}>
                <View style={styles.chatContainer}>
                    <Content />
                </View>
                <View>
                    <Chatbox />
                </View>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    viewContainer: {
        flex: 1,
    },
    chatContainer: {
        flex: 1,
    },

});
